import os
from typing import Tuple

from pipeline_train import (
    path_inference_cfg,
    path_list_col_high_corr,
    path_list_col_heavy_nan,
    # path_dict_col_category,
    path_list_col_nominal,
    path_list_col_numeric,
    path_model_pipeline
)
from src.utils import *
from src.data_preprocessing import *
from src.model_training import preprocess_nominal

file_name_features_inference = file_name_fmt_csv.replace("{prefix}", "features_predict")
file_name_inference = file_name_fmt_csv.replace("{prefix}", "prediction")

path_application_test = os.path.join(DIR_RAW_DATA, 'application_test.csv')
path_bereau = os.path.join(DIR_RAW_DATA, 'bureau.csv')
path_bereau_balance = os.path.join(DIR_RAW_DATA, 'bureau_balance.csv')
path_previous_application = os.path.join(DIR_RAW_DATA, 'previous_application.csv')
path_pos_cash_balance = os.path.join(DIR_RAW_DATA, 'POS_CASH_balance.csv')
path_installment_payments = os.path.join(DIR_RAW_DATA, 'installments_payments.csv')
path_credit_card_balance = os.path.join(DIR_RAW_DATA, 'credit_card_balance.csv')

path_inference_features = os.path.join(DIR_PREPROCESS_DATA_INFERENCE, file_name_features_inference)
path_inference = os.path.join(DIR_INFERENCE_OUTPUT, file_name_inference)

# get logger
logger = get_logger(__name__, log_level='DEBUG')


def read_training_cfg() -> Tuple[str, float]:
    inference_cfg = load_dict(path_inference_cfg, logger=logger)

    train_version = inference_cfg['version']
    prob_threshold = inference_cfg['probability_threshold']

    return train_version, prob_threshold


def read_raw_data() -> List[pd.DataFrame]:
    # read data
    df_application_test = pd.read_csv(path_application_test)
    df_bereau = pd.read_csv(path_bereau)
    df_bereau_balance = pd.read_csv(path_bereau_balance)
    df_previous_application = pd.read_csv(path_previous_application)
    df_pos_cash_balance = pd.read_csv(path_pos_cash_balance)
    df_installment_payments = pd.read_csv(path_installment_payments)
    # df_credit_card_balance = pd.read_csv(path_credit_card_balance)

    return df_application_test, df_bereau, df_bereau_balance, df_previous_application, df_pos_cash_balance, df_installment_payments


def preprocess_step(version: str) -> Tuple[str, Dict[str, str]]:
    logger.debug("STEP PROCESSING")

    # read data
    df_application_test, df_bereau, df_bereau_balance, df_previous_application, \
        df_pos_cash_balance, df_installment_payments = read_raw_data()
    
    # load preprocess artifacts
    list_col_high_corr = load_list(path_list_col_high_corr.format(suffix=version), logger=logger)
    list_col_heavy_nan = load_list(path_list_col_heavy_nan.format(suffix=version), logger=logger)
    list_col_nominal_selected = load_list(path_list_col_nominal.format(suffix=version), logger=logger)
    list_col_numeric_selected = load_list(path_list_col_numeric.format(suffix=version), logger=logger)

    # preprocess data
    df_application_test, _ = preprocess_application_data(
        df_application_test, list_col_name_numeric=list(), list_col_name_nominal=list(),
        list_col_to_drop=list_col_to_drop, list_col_contain_outlier=list()
    )
    df_bureau_bal_ref = preprocess_df_bureau_ref(df_bereau_balance)
    df_id_ref = df_application_test.loc[:, [col_name_id_appl]]
    df_bereau = preprocess_df_bereau(df_id_ref, df_bereau, df_bureau_bal_ref)
    df_previous_application_prep = preprocess_df_prev_application_data(df_previous_application)

    # feature engineering
    df_bereau_features = merge_bereau_feature(df_id_ref, df_bereau)
    df_prev_appl_features = merge_prev_app_features(df_id_ref, df_previous_application_prep)
    df_pos_bal_features = merge_pos_cash_bal(df_id_ref, df_pos_cash_balance)
    df_inst_payment_features = merge_installment_payment(df_id_ref, df_installment_payments)

    # data aggregation
    list_dfs = [df_application_test, df_bereau_features, df_prev_appl_features, df_pos_bal_features, df_inst_payment_features]
    df_test_features = reduce(lambda left, right: pd.merge(left, right, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner'), list_dfs)

    # remove correlated columns (numeric)
    df_test_features.drop(columns=list_col_high_corr, inplace=True)

    # remove heavy NaN col
    df_test_features.drop(columns=list_col_heavy_nan, inplace=True)

    # filter relevant col
    df_test_features.set_index(col_name_id_appl, inplace=True)
    df_test_features = df_test_features.loc[:, [*list_col_nominal_selected, *list_col_numeric_selected]]

    logger.debug("Inference features shape --> {}".format(df_test_features.shape))

    # create directory
    cre8_dir(DIR_PREPROCESS_DATA_INFERENCE, logger=logger)

    # file path
    inference_suffix = cre8_file_suffix()

    path_inference_features_save = path_inference_features.format(suffix=inference_suffix)

    # save file
    save_csv_file(df_test_features, save_path=path_inference_features_save, logger=logger)

    dict_paths = dict(
        path_features_inference=path_inference_features_save
    )

    return inference_suffix, dict_paths


def inference_step(version: str, inference_version:  str, prob_threshold: float, paths_: Dict[str, str]) -> None:
    logger.debug("STEP TRAINING")

    # unzip paths from previous step
    path_inference_features_save = paths_['path_features_inference']

    # load files
    df_features_inference = load_csv_file(path_inference_features_save, logger=logger)
    model_pipeline = load_model(load_path=path_model_pipeline.format(suffix=version), logger=logger)
    list_col_nominal_selected = load_list(path_list_col_nominal.format(suffix=version), logger=logger)

    df_features_inference = preprocess_nominal(df_features_inference, list_nominal_col=list_col_nominal_selected, 
                                          return_objects_for_encoder=False)
    
    # prediction
    arr_prob_prediction = model_pipeline.predict_proba(df_features_inference)
    arr_1_prob_prediction = arr_prob_prediction[:, 1]
    prediction_w_threshold = arr_1_prob_prediction > prob_threshold
    arr_class_prediction = prediction_w_threshold.astype(int)

    df_prediction = pd.DataFrame({'pred': arr_class_prediction, 'prob': arr_1_prob_prediction}, index=df_features_inference.index)

    # create directory
    cre8_dir(DIR_INFERENCE_OUTPUT, logger=logger)

    # file path
    path_inference_output_save = path_inference.format(suffix=inference_version)

    # export data
    save_csv_file(df_prediction, save_path=path_inference_output_save, logger=logger)


def main():
    logger.info("Start inference pipeline")

    # step 0: read training cfg
    train_version, probability_threshold = read_training_cfg()

    # step 1: preprocessing data
    inference_version, paths_1 = preprocess_step(version=train_version)

    # step 2: data inference
    inference_step(version=train_version, inference_version=inference_version, prob_threshold=probability_threshold,
                   paths_=paths_1)

    logger.info("Completed inference pipeline")


if __name__ == "__main__":
    main()
