from sklearn.model_selection import train_test_split

from src.data_preprocessing import *
from src.model_training import *
from src.model_evaluation import roc_auc_score, accuracy_report
from src.utils import *
from src.utils import (
    DIR_RAW_DATA,
    DIR_PREPROCESS_DATA_ARTIFACTS,
    DIR_PREPROCESS_DATA_MODELING_TRAIN,
    DIR_PREPROCESS_DATA_MODELING_TEST
)

TRAIN_SIZE = 0.9
RANDOM_STATE = 99

prefix_train_features = "features_train"
prefix_eval_features = "features_test"
prefix_train_target = "target_train"
prefix_eval_target = "target_test"
prefix_list_col_numeric = "list_col_numeric"
prefix_list_col_nominal = "list_col_nominal"
prefix_list_col_high_corr = "list_col_high_corr"
prefix_list_col_heavy_nan = "list_col_heavy_nan"
prefix_dict_selected_category = "dict_col_selected_cat"

# join path
path_application_train = os.path.join(DIR_RAW_DATA, 'application_train.csv')
path_bereau = os.path.join(DIR_RAW_DATA, 'bureau.csv')
path_bereau_balance = os.path.join(DIR_RAW_DATA, 'bureau_balance.csv')
path_previous_application = os.path.join(DIR_RAW_DATA, 'previous_application.csv')
path_pos_cash_balance = os.path.join(DIR_RAW_DATA, 'POS_CASH_balance.csv')
path_installment_payments = os.path.join(DIR_RAW_DATA, 'installments_payments.csv')
path_credit_card_balance = os.path.join(DIR_RAW_DATA, 'credit_card_balance.csv')

# get logger
logger = get_logger(__name__, log_level='DEBUG')


def read_raw_data() -> List[pd.DataFrame]:
    # read data
    df_application_train = pd.read_csv(path_application_train)
    df_bereau = pd.read_csv(path_bereau)
    df_bereau_balance = pd.read_csv(path_bereau_balance)
    df_previous_application = pd.read_csv(path_previous_application)
    df_pos_cash_balance = pd.read_csv(path_pos_cash_balance)
    df_installment_payments = pd.read_csv(path_installment_payments)
    # df_credit_card_balance = pd.read_csv(path_credit_card_balance)

    return df_application_train, df_bereau, df_bereau_balance, df_previous_application, df_pos_cash_balance, df_installment_payments


def preprocess_step() -> Tuple[str, Dict[str]]:
    logger.debug("STEP PROCESSING")

    # read data
    df_application_train, df_bereau, df_bereau_balance, df_previous_application, df_pos_cash_balance, df_installment_payments = read_raw_data()

    # column description
    list_col_name_numeric = list_col_numeric.copy() # save later
    list_col_name_nominal = list_col_nominal.copy() # save later

    # preprocess data (save dict_category)
    df_application_train, dict_category = preprocess_application_data(df_application_train, list_col_name_numeric, list_col_name_nominal,
                                                                      list_col_to_drop=list_col_to_drop, list_col_contain_outlier=list_col_contain_outlier)
    df_bureau_bal_ref = preprocess_df_bureau_ref(df_bereau_balance)
    df_id_ref = df_application_train.loc[:, [col_name_id_appl]]
    df_bereau = preprocess_df_bereau(df_id_ref, df_bereau, df_bureau_bal_ref)
    df_previous_application_prep = preprocess_df_prev_application_data(df_previous_application)

    # feature engineering
    df_bereau_features = merge_bereau_feature(df_id_ref, df_bereau, 
                                              list_col_name_nominal=list_col_name_nominal, list_col_name_numeric=list_col_name_numeric)
    df_prev_appl_features = merge_prev_app_features(df_id_ref, df_previous_application_prep,
                                                    list_col_name_nominal=list_col_name_nominal, list_col_name_numeric=list_col_name_numeric)
    df_pos_bal_features = merge_pos_cash_bal(df_id_ref, df_pos_cash_balance,
                                             list_col_name_nominal=list_col_name_nominal, list_col_name_numeric=list_col_name_numeric)
    df_inst_payment_features = merge_installment_payment(df_id_ref, df_installment_payments,
                                                         list_col_name_nominal=list_col_name_nominal, list_col_name_numeric=list_col_name_numeric)
    
    # data aggregation
    list_dfs = [df_application_train, df_bereau_features, df_prev_appl_features, df_pos_bal_features, df_inst_payment_features]
    df_train_features = reduce(lambda left, right: pd.merge(left, right, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner'), list_dfs)

    # remove correlated columns (numeric)
    df_corr = calculate_correlation_matrix_with_nan(df_train_features, list_col_name_numeric=list_col_name_numeric)
    list_col_high_corr = remove_correlate_features(df_corr) # save obj

    df_train_features.drop(columns=list_col_high_corr, inplace=True)
    list_col_name_numeric = [col for col in list_col_name_numeric if col not in list_col_high_corr]

    # remove heavy NaN col
    df_train_features, list_col_heavy_nan = remove_heavy_nan_col(df_train_features, list_col_name_nominal=list_col_name_nominal, 
                                                                 list_col_name_numeric=list_col_name_numeric)
    
    # train-validate-test split
    Y = df_train_features[col_name_target]
    X = df_train_features.drop(columns=[col_name_id_appl, col_name_target])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=TRAIN_SIZE, stratify=Y, shuffle=True)

    logger.debug("Training features shape --> {}, Training target shape --> {}".format(X_train.shape, Y_train.shape))
    logger.debug("Evaluation features shape --> {}, Evaluation target shape --> {}".format(X_test.shape, Y_test.shape))
    
    # create directory
    cre8_dir(DIR_PREPROCESS_DATA_ARTIFACTS, logger=logger)
    cre8_dir(DIR_PREPROCESS_DATA_MODELING_TRAIN, logger=logger)
    cre8_dir(DIR_PREPROCESS_DATA_MODELING_TEST, logger=logger)

    # file path
    file_suffix = cre8_file_suffix()

    file_name_features_train = file_name_fmt_csv.format(prefix=prefix_train_features, suffix=file_suffix)
    file_name_features_eval = file_name_fmt_csv.format(prefix=prefix_eval_features, suffix=file_suffix)
    file_name_target_train = file_name_fmt_array.format(prefix=prefix_train_target, suffix=file_suffix)
    file_name_target_eval = file_name_fmt_array.format(prefix=prefix_eval_target, suffix=file_suffix)
    file_name_list_numeric = file_name_fmt_list.format(prefix=prefix_list_col_numeric, suffix=file_suffix)
    file_name_list_nominal = file_name_fmt_list.format(prefix=prefix_list_col_nominal, suffix=file_suffix)
    file_name_list_high_corr = file_name_fmt_list.format(prefix=prefix_list_col_high_corr, suffix=file_suffix)
    file_name_list_heavy_nan = file_name_fmt_list.format(prefix=prefix_list_col_heavy_nan, suffix=file_suffix)
    file_name_dict_category = file_name_fmt_dict.format(prefix=prefix_dict_selected_category, suffix=file_suffix)

    path_modeling_features_train = os.path.join(DIR_PREPROCESS_DATA_MODELING_TRAIN, file_name_features_train)
    path_modeling_features_eval = os.path.join(DIR_PREPROCESS_DATA_MODELING_TEST, file_name_features_eval)
    path_modeling_target_train = os.path.join(DIR_PREPROCESS_DATA_MODELING_TRAIN, file_name_target_train)
    path_modeling_target_eval = os.path.join(DIR_PREPROCESS_DATA_MODELING_TEST, file_name_target_eval)
    path_list_col_numeric = os.path.join(DIR_PREPROCESS_DATA_ARTIFACTS, file_name_list_numeric)
    path_list_col_nominal = os.path.join(DIR_PREPROCESS_DATA_ARTIFACTS, file_name_list_nominal)
    path_list_col_high_corr = os.path.join(DIR_PREPROCESS_DATA_ARTIFACTS, file_name_list_high_corr)
    path_list_col_heavy_nan = os.path.join(DIR_PREPROCESS_DATA_ARTIFACTS, file_name_list_heavy_nan)
    path_dict_col_category = os.path.join(DIR_PREPROCESS_DATA_ARTIFACTS, file_name_dict_category)

    # save files and artifacts
    ## csv file
    save_csv_file(X_train, save_path=path_modeling_features_train, logger=logger)
    save_csv_file(X_test, save_path=path_modeling_features_eval, logger=logger)

    ## nd array
    save_ndarray(Y_train, save_path=path_modeling_target_train, logger=logger)
    save_ndarray(Y_test, save_path=path_modeling_target_eval, logger=logger)

    ## list
    save_list(list_col_name_numeric, save_path=path_list_col_numeric, logger=logger)
    save_list(list_col_name_nominal, save_path=path_list_col_nominal, logger=logger)
    save_list(list_col_high_corr, save_path=path_list_col_high_corr, logger=logger)
    save_list(list_col_heavy_nan, save_path=path_list_col_heavy_nan, logger=logger)

    ## dict
    save_dict(dict_category, save_path=path_dict_col_category, logger=logger)

    dict_paths = dict(
        path_features_train=path_modeling_features_train, 
        path_features_eval=path_modeling_features_eval,
        path_target_train=path_modeling_target_train,
        path_target_eval=path_modeling_target_eval,
        path_numeric_cols=path_list_col_numeric,
        path_nominal_cols=path_list_col_nominal,
        path_high_corr_cols=path_list_col_high_corr,
        path_heavy_nan_cols=path_list_col_heavy_nan,
        path_selected_categories=path_dict_col_category
    )

    return file_suffix, dict_paths


def model_training_step(version: str, paths_: Dict[str]) -> None:
    logger.debug("STEP TRAINING")

    # unzip paths from previous step
    path_modeling_features_train = paths_['path_features_train']
    path_modeling_target_train = paths_['path_target_train']
    path_dict_col_category = paths_['path_selected_categories']
    path_list_col_numeric = paths_['path_numeric_cols']
    path_list_col_nominal = paths_['path_nominal_cols']

    # read processed data (from previous step)
    df_features_train = load_csv_file(path_modeling_features_train, logger=logger)
    arr_target_train = load_ndarray(path_modeling_target_train, logger=logger)
    dict_categories_ = load_dict(path_dict_col_category, logger=logger)
    list_col_name_numeric = load_list(path_list_col_numeric, logger=logger)
    list_col_name_nominal = load_list(path_list_col_nominal, logger=logger)

    arr_target_train = pd.Series(arr_target_train, index=df_features_train.index)

    # final preprocess on nominal col
    df_features_train, list_col_object, categories_ = preprocess_nominal(df_features_train, 
                                                                         list_nominal_col=list_col_name_nominal, 
                                                                         dict_category=dict_categories_,
                                                                         return_objects_for_encoder=True)
    
    # create pipeline object
    pipeline = cre8_sklearn_pipeline(arr_target_train, 
                                     list_col_cat=list_col_object, list_col_num=list_col_name_numeric, 
                                     list_categories_=categories_, logger=logger)
    model = pipeline['model']
    if type(model).__name__ == 'XGBClassifier':
        sample_weight = compute_sample_weight(class_weight="balanced", y=arr_target_train)
        pipeline.fit(df_features_train, arr_target_train, model__sample_weight=sample_weight)
    else:
        pipeline.fit(df_features_train, arr_target_train)

    # evaluation (on Training Set)
    arr_prob_train_pred = pipeline.predict_proba(df_features_train)
    p_train_pred = arr_prob_train_pred[:, 1]
    auc_score = roc_auc_score(arr_target_train, p_train_pred)
    


def model_evaluation_step(version: str) -> None:
    pass


def main():
    logger.info("Start training pipeline")

    # step 1)
    training_version = preprocess_step()
    # step 2)
    model_training_step(training_version)
    # step 3)
    model_evaluation_step(training_version)

    logger.info("Completed training pipeline")


if __name__ == "__main__":
    main()