from sklearn.model_selection import train_test_split

from src.data_preprocessing import *
from src.model_training import *
from src.model_evaluation import roc_auc_score, accuracy_report
from src.utils import *
from src.utils import (
    DIR_RAW_DATA,
    DIR_PREPROCESS_DATA_ARTIFACTS,
    DIR_PREPROCESS_DATA_MODELING_TRAIN,
    DIR_PREPROCESS_DATA_MODELING_TEST,
    DIR_MODEL
)

TRAIN_SIZE = 0.9
RANDOM_STATE = 99

file_name_features_train = file_name_fmt_csv.replace("{prefix}", "features_train")
file_name_features_eval = file_name_fmt_csv.replace("{prefix}", "features_test")
file_name_target_train = file_name_fmt_array.replace("{prefix}", "target_train")
file_name_target_eval = file_name_fmt_array.replace("{prefix}", "target_test")
file_name_list_numeric = file_name_fmt_list.replace("{prefix}", "list_col_numeric")
file_name_list_nominal = file_name_fmt_list.replace("{prefix}", "list_col_nominal")
file_name_list_high_corr = file_name_fmt_list.replace("{prefix}", "list_col_high_corr")
file_name_list_heavy_nan = file_name_fmt_list.replace("{prefix}", "list_col_heavy_nan")
file_name_dict_category = file_name_fmt_dict.replace("{prefix}", "dict_col_selected_cat")
file_name_model_pipeline = file_name_fmt_list.replace("{prefix}", "model_pipeline")

# join path
dir_inference_cfg = os.path.join(DIR_CONFIG, 'inference')

path_application_train = os.path.join(DIR_RAW_DATA, 'application_train.csv')
path_bereau = os.path.join(DIR_RAW_DATA, 'bureau.csv')
path_bereau_balance = os.path.join(DIR_RAW_DATA, 'bureau_balance.csv')
path_previous_application = os.path.join(DIR_RAW_DATA, 'previous_application.csv')
path_pos_cash_balance = os.path.join(DIR_RAW_DATA, 'POS_CASH_balance.csv')
path_installment_payments = os.path.join(DIR_RAW_DATA, 'installments_payments.csv')
path_credit_card_balance = os.path.join(DIR_RAW_DATA, 'credit_card_balance.csv')
path_log_version = os.path.join(DIR_CONFIG, 'version', 'pipeline_performance.log')
path_inference_cfg = os.path.join(dir_inference_cfg, 'inference_config.json')

path_modeling_features_train = os.path.join(DIR_PREPROCESS_DATA_MODELING_TRAIN, file_name_features_train)
path_modeling_features_eval = os.path.join(DIR_PREPROCESS_DATA_MODELING_TEST, file_name_features_eval)
path_modeling_target_train = os.path.join(DIR_PREPROCESS_DATA_MODELING_TRAIN, file_name_target_train)
path_modeling_target_eval = os.path.join(DIR_PREPROCESS_DATA_MODELING_TEST, file_name_target_eval)
path_list_col_numeric = os.path.join(DIR_PREPROCESS_DATA_ARTIFACTS, file_name_list_numeric)
path_list_col_nominal = os.path.join(DIR_PREPROCESS_DATA_ARTIFACTS, file_name_list_nominal)
path_list_col_high_corr = os.path.join(DIR_PREPROCESS_DATA_ARTIFACTS, file_name_list_high_corr)
path_list_col_heavy_nan = os.path.join(DIR_PREPROCESS_DATA_ARTIFACTS, file_name_list_heavy_nan)
path_dict_col_category = os.path.join(DIR_PREPROCESS_DATA_ARTIFACTS, file_name_dict_category)
path_model_pipeline = os.path.join(DIR_MODEL, file_name_model_pipeline)

# get logger
logger = get_logger(__name__, log_level='DEBUG')
logger_version = get_logger("Training Version", path_log_filename=path_log_version, log_level='INFO')


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


def preprocess_step() -> Tuple[str, Dict[str, str]]:
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

    path_modeling_features_train_save = path_modeling_features_train.format(suffix=file_suffix)
    path_modeling_features_eval_save = path_modeling_features_eval.format(suffix=file_suffix)
    path_modeling_target_train_save = path_modeling_target_train.format(suffix=file_suffix)
    path_modeling_target_eval_save = path_modeling_target_eval.format(suffix=file_suffix)
    path_list_col_numeric_save = path_list_col_numeric.format(suffix=file_suffix)
    path_list_col_nominal_save = path_list_col_nominal.format(suffix=file_suffix)
    path_list_col_high_corr_save = path_list_col_high_corr.format(suffix=file_suffix)
    path_list_col_heavy_nan_save = path_list_col_heavy_nan.format(suffix=file_suffix)
    path_dict_col_category_save = path_dict_col_category.format(suffix=file_suffix)

    # save files and artifacts
    ## csv file
    save_csv_file(X_train, save_path=path_modeling_features_train_save, logger=logger)
    save_csv_file(X_test, save_path=path_modeling_features_eval_save, logger=logger)

    ## nd array
    save_ndarray(Y_train, save_path=path_modeling_target_train_save, logger=logger)
    save_ndarray(Y_test, save_path=path_modeling_target_eval_save, logger=logger)

    ## list
    save_list(list_col_name_numeric, save_path=path_list_col_numeric_save, logger=logger)
    save_list(list_col_name_nominal, save_path=path_list_col_nominal_save, logger=logger)
    save_list(list_col_high_corr, save_path=path_list_col_high_corr_save, logger=logger)
    save_list(list_col_heavy_nan, save_path=path_list_col_heavy_nan_save, logger=logger)

    ## dict
    save_dict(dict_category, save_path=path_dict_col_category_save, logger=logger)

    dict_paths = dict(
        path_features_train=path_modeling_features_train_save, 
        path_features_eval=path_modeling_features_eval_save,
        path_target_train=path_modeling_target_train_save,
        path_target_eval=path_modeling_target_eval_save,
        path_numeric_cols=path_list_col_numeric_save,
        path_nominal_cols=path_list_col_nominal_save,
        path_high_corr_cols=path_list_col_high_corr_save,
        path_heavy_nan_cols=path_list_col_heavy_nan_save,
        path_selected_categories=path_dict_col_category_save
    )

    return file_suffix, dict_paths


def model_training_step(version: str, paths_: Dict[str, str]) -> Dict[str, str]:
    logger.debug("STEP TRAINING")

    # unzip paths from previous step
    path_modeling_features_train = paths_['path_features_train']
    path_modeling_target_train = paths_['path_target_train']
    path_dict_col_category = paths_['path_selected_categories']
    path_list_col_numeric = paths_['path_numeric_cols']
    path_list_col_nominal = paths_['path_nominal_cols']

    # read data and related files (from previous step)
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
    logger.debug("Training AUC score = {}".format(round(auc_score, 4)))

    # create directory
    cre8_dir(DIR_MODEL, logger=logger)

    # file name and path
    path_model_pipeline_save = path_model_pipeline.format(suffix=version)

    # save model pipeline
    save_model(model=pipeline, save_path=path_model_pipeline_save, logger=logger)

    dict_path = dict(
        path_model_pipeline=path_model_pipeline_save,
        path_nominal_cols=path_list_col_nominal,
        path_features_eval=paths_['path_features_eval'],
        path_target_eval=paths_['path_target_eval']
    )

    return dict_path


def model_evaluation_step(version: str, paths_: Dict[str, str]) -> None:
    logger.debug("STEP EVALUATE")

    # unzip path from previous step
    path_modeling_features_eval = paths_['path_features_eval']
    path_modeling_target_eval = paths_['path_target_eval']
    path_model_pipeline = paths_['path_model_pipeline']
    path_list_col_nominal = paths_['path_nominal_cols']

    # read data and related files (from previous step)
    df_features_eval = load_csv_file(path_modeling_features_eval, logger=logger)
    arr_target_eval = load_ndarray(path_modeling_target_eval, logger=logger)
    model_pipeline = load_model(load_path=path_model_pipeline, logger=logger)
    list_col_name_nominal = load_list(path_list_col_nominal, logger=logger)

    arr_target_eval = pd.Series(arr_target_eval, index=df_features_eval.index)

    df_features_eval = preprocess_nominal(df_features_eval, list_nominal_col=list_col_name_nominal, 
                                          return_objects_for_encoder=False)
    
    # evaluation (on Eval Set)
    arr_eval_pred = model_pipeline.predict(df_features_eval)
    arr_prob_eval_pred = model_pipeline.predict_proba(df_features_eval)

    p_eval_pred = arr_prob_eval_pred[:, 1]
    auc_score = roc_auc_score(arr_target_eval, p_eval_pred)
    dict_accuracy_score = accuracy_report(y_true=arr_target_eval, y_pred=arr_eval_pred)

    confusion_matrix_eval = dict_accuracy_score['confusion_matrix']
    accuracy_eval = dict_accuracy_score['accuracy_score']
    precision_eval = dict_accuracy_score['precision_score']
    recall_eval = dict_accuracy_score['recall_score']
    f1_eval = dict_accuracy_score['f1_score']

    logger_version.info("Training version \"{}\", AUC score = {}".format(version, round(auc_score, 4)))

    logger.debug("Evaluation AUC score = {}".format(round(auc_score, 4)))

    logger.debug("Confusion Matrix : \n{}".format(confusion_matrix_eval))
    logger.debug("Accuracy Score : {0:.2f}".format(accuracy_eval))
    logger.debug("Precision Score : {0:.2f}".format(precision_eval))
    logger.debug("Recall Score : {0:.2f}".format(recall_eval))
    logger.debug("F1 Score : {0:.2f}".format(f1_eval))


def inference_config(version: str):
    logger.debug("WRITE CONFIG")   

    config = {
        "version": version,
        "probability_threshold": 0.04
    }

    cre8_dir(dir_inference_cfg, logger=logger)
    save_dict(config, save_path=path_inference_cfg, logger=logger)


def main():
    logger.info("Start training pipeline")

    # step 1: preprocessing data
    training_version, paths_1 = preprocess_step()

    # step 2: training model
    path_2 = model_training_step(version=training_version, paths_=paths_1)

    # step 3: model evaluation
    model_evaluation_step(version=training_version, paths_=path_2)

    # step 4: inference config
    inference_config(version=training_version)

    logger.info("Completed training pipeline")


if __name__ == "__main__":
    main()
