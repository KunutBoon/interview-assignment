import os
import math
from typing import List, Dict, Union, Optional, Tuple
from functools import reduce

import numpy as np
import pandas as pd


col_name_id_appl = 'SK_ID_CURR'
col_name_target = 'TARGET'

list_col_nominal = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE',
    'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
    'FLAG_PHONE', 'FLAG_EMAIL', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
    'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_2', 
    'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 
    'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'
]
list_col_numeric = [
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 
    'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'CNT_FAM_MEMBERS', 'HOUR_APPR_PROCESS_START', 'EXT_SOURCE_1', 
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 
    'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 
    'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 
    'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 
    'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 
    'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 
    'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE', 
    'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 
    'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 
    'AMT_REQ_CREDIT_BUREAU_YEAR'
]
list_col_to_drop = [
    'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'FONDKAPREMONT_MODE', 'EMERGENCYSTATE_MODE',  'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_2', 
    'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7',  'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 
    'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 
    'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK'
]
list_col_contain_outlier = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'YEARS_BEGINEXPLUATATION_AVG',  'ENTRANCES_AVG', 'YEARS_BUILD_MODE', 
    'OBS_60_CNT_SOCIAL_CIRCLE', 'OBS_30_CNT_SOCIAL_CIRCLE'
]

file_name_fmt_array = "{prefix}_{suffix}.npy"
file_name_fmt_list = "{prefix}_{suffix}.sav"
file_name_fmt_dict = "{prefix}_{suffix}.json"
file_name_fmt_csv = "{prefix}_{suffix}.csv"


def filter_high_freq_category(df_app: pd.DataFrame, list_col_name_nominal: List[str], threshold: Union[int, float] = 0.03) -> Dict[str, List[str]]:
    dict_nominal_category = dict()
    
    if type(threshold) == float:
        n_min_records = round(df_app.shape[0] * threshold, 0)
    else:
        n_min_records = threshold

    for col in list_col_name_nominal:
        value_counts = df_app[col].value_counts(dropna=True)
        filter_high_freq = value_counts.gt(n_min_records)
        list_category_high_freq = value_counts[filter_high_freq].index.tolist()
        dict_nominal_category[col] = list_category_high_freq

    return dict_nominal_category


def handle_outliers(df: pd.DataFrame, col_name: str, drop_outlier: bool = False) -> pd.DataFrame:
    df = df.copy()
    arr_col_stats = df[col_name].describe()

    q1 = arr_col_stats.loc['25%']
    q3 = arr_col_stats.loc['75%']
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    if drop_outlier:
        filter_non_outlier = df[col_name].between(lower, upper)
        df = df.loc[filter_non_outlier, :]
    else:
        df.loc[df[col_name] < lower, col_name] = lower
        df.loc[df[col_name] > upper, col_name] = upper
    
    return df


def update_remove_cols(list_col: List[str], list_remove_col: List[str]) -> List[str]:
    for col in list_remove_col:
        if col in list_col:
            list_col.remove(col)


def preprocess_application_data(df_app: pd.DataFrame, list_col_name_numeric: List[str], list_col_name_nominal: List[str],
                                list_col_to_drop: List[str], list_col_contain_outlier: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    df = df_app.copy()

    df.drop(columns=list_col_to_drop, inplace=True)
    update_remove_cols(list_col=list_col_name_numeric, list_remove_col=list_col_to_drop)
    update_remove_cols(list_col=list_col_name_nominal, list_remove_col=list_col_to_drop)

    dict_category = filter_high_freq_category(df, list_col_name_nominal=list_col_name_nominal, threshold=.02)
    
    # convert dtypes
    for col in list_col_name_nominal:
        categories_ = dict_category[col]
        if len(categories_) == 1:
            df.drop(columns=[col], inplace=True)
            dict_category.pop(col)
            update_remove_cols(list_col=list_col_name_nominal, list_remove_col=[col, ])
    
    # remove outlier
    for col in list_col_contain_outlier:
        bool_remove_outliers_ = col == 'AMT_INCOME_TOTAL'
        handle_outliers(df, col_name=col, drop_outlier=bool_remove_outliers_)

    # remove weird values in DAYS_EMPLOYED col
    df['DAYS_EMPLOYED'] = np.minimum(0, df['DAYS_EMPLOYED'])

    return df, dict_category


def calculate_correlation_matrix_with_nan(df: pd.DataFrame, list_col_name_numeric: List[str]) -> pd.DataFrame:
    df_corr_matrix = pd.DataFrame(index=list_col_name_numeric, columns=list_col_name_numeric)
    for col_1 in list_col_name_numeric:
        for col_2 in list_col_name_numeric:
            if col_1 == col_2:
                corr_1_2 = 1
            else:
                df_col_1_2 = df.loc[:, [col_1, col_2]]
                n_null_rows = df_col_1_2.isnull().any(axis=1).sum()
                if n_null_rows > 0:
                    df_col_1_2.dropna(axis=0, how='any', inplace=True)
                df_corr_1_2 = df_col_1_2.corr()
                corr_1_2 = df_corr_1_2.loc[col_1, col_2]
                # update lower triangle
                df_corr_matrix.loc[col_2, col_1] = corr_1_2
            
            # update upper triangle / diagonal
            df_corr_matrix.loc[col_1, col_2] = corr_1_2
    
    df_corr_matrix = df_corr_matrix.astype(float)
    
    return df_corr_matrix


def list_heavy_nan_col(df: pd.DataFrame, threshold: float = 0.4) -> List[str]:
    n_row, _ = df.shape
    n_row_max_null_allow = math.ceil(n_row * threshold)
    arr_count_nan = df.isnull().sum()
    list_col_heavy_nan = arr_count_nan[arr_count_nan > n_row_max_null_allow].index.tolist()

    return list_col_heavy_nan


def remove_heavy_nan_col(df: pd.DataFrame, threshold: float = 0.4, list_col_name_numeric: Optional[List[str]] = None, 
                         list_col_name_nominal: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    list_col_heavy_nan = list_heavy_nan_col(df, threshold=threshold)
    df = df.drop(columns=list_col_heavy_nan)

    # update column definition
    for col in list_col_heavy_nan:
        if (list_col_name_nominal is not None) and (col in list_col_name_nominal):
            list_col_name_nominal.remove(col)
        if (list_col_name_numeric is not None) and (col in list_col_name_numeric):
            list_col_name_numeric.remove(col)

    return df, list_col_heavy_nan


def remove_correlate_features(df_corr: pd.DataFrame, threshold: float = 0.8, 
                              list_col_numeric: Optional[List[str]] = None, list_col_remove: List[str] = list()) -> List[str]:
    if list_col_numeric is None:
        list_col_numeric = df_corr.columns.tolist()
    
    col_name = list_col_numeric.pop(0)

    # check correlation
    list_col_compare = list(filter(lambda x: x != col_name, list_col_numeric))
    arr_corr_values = df_corr.loc[col_name, list_col_compare]
    list_correlated_col = arr_corr_values[arr_corr_values > threshold].index.tolist()

    # update list remove col if correlated
    if len(list_correlated_col) > 0:
        for col in list_correlated_col:
            if col in list_col_numeric:
                list_col_numeric.remove(col)
            if col not in list_col_remove:
                list_col_remove.append(col)
    
    if len(list_col_numeric) > 1:
        remove_correlate_features(df_corr, threshold=threshold, list_col_numeric=list_col_numeric, list_col_remove=list_col_remove)
    
    return list_col_remove


def preprocess_df_bureau_ref(df_bal: pd.DataFrame, n_years: int = 6) -> pd.DataFrame:
    n_months_minimum = -12 * n_years

    # feature engineering: is_dpd
    list_status_dpd = ('1', '2', '3', '4', '5')
    df_bal['is_month_dpd'] = df_bal['STATUS'].isin(list_status_dpd)

    # aggregation : latest info
    idx_latest = df_bal.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].idxmax()
    df_latest_info = df_bal.loc[idx_latest, ['SK_ID_BUREAU', 'MONTHS_BALANCE', 'STATUS']]
    df_latest_info['status_ref'] = np.where(df_latest_info['STATUS'].eq('C'), 'Closed',
                                            np.where(df_latest_info['STATUS'].eq('X'), 'Unknown', 'Active'))
    df_latest_info.drop(columns='STATUS', inplace=True)
    
    # aggregation: historical dpd for each bureau
    df_historical_dpd = df_bal.loc[df_bal['MONTHS_BALANCE'].gt(n_months_minimum)].groupby('SK_ID_BUREAU')['is_month_dpd'].sum().reset_index()
    df_bureau_crossref = df_latest_info.merge(df_historical_dpd, left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU', how='inner')
    df_bureau_crossref = df_bureau_crossref.rename(columns={'is_month_dpd': 'count_month_dpd', 'MONTHS_BALANCE': 'month_latest_update'})

    return df_bureau_crossref


def preprocess_df_bereau(df_id_ref: pd.DataFrame, df_bereau: pd.DataFrame, df_bereau_ref: pd.DataFrame, n_years: int = 6) -> pd.DataFrame:
    n_days_minimum = -365 * n_years
    
    # scope relevant information (by ID)
    df_bereau = df_id_ref.merge(df_bereau, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner')

    filter_currency = df_bereau['CREDIT_CURRENCY'].eq('currency 1')
    filter_incorrect_update_date = df_bereau['DAYS_CREDIT_UPDATE'] <= 0
    filter_new_update_info = df_bereau['DAYS_CREDIT_UPDATE'] > n_days_minimum

    df_bereau = df_bereau.loc[filter_currency & filter_incorrect_update_date & filter_new_update_info, :]

    # cross-reference
    df_bereau = df_bereau.merge(df_bereau_ref, left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU', how='left')
    df_bereau['credit_status'] = np.where(df_bereau['status_ref'].notna(), df_bereau['status_ref'], df_bereau['CREDIT_ACTIVE'])

    # filter out incorrect DAYS_CREDIT_ENDDATE
    filter_out_incorrect_active_enddate = (df_bereau['credit_status'] == 'Active') & (df_bereau['DAYS_CREDIT_ENDDATE'] < 0)
    filter_out_incorrect_closed_enddate = (df_bereau['credit_status'] == 'Closed') & (df_bereau['DAYS_CREDIT_ENDDATE'] >= 0)
    filter_other_status = df_bereau['credit_status'].isin(('Active', 'Closed'))
    df_bereau = df_bereau.loc[filter_out_incorrect_active_enddate | filter_out_incorrect_closed_enddate | filter_other_status, :]

    return df_bereau


def merge_bereau_feature(df_id_ref: pd.DataFrame, df_bereau_prep: pd.DataFrame, list_col_name_nominal: Optional[List[str]] = None,
                         list_col_name_numeric: Optional[list[str]] = None) -> pd.DataFrame:
    list_bereau_col_select = ['SK_ID_CURR', 'SK_ID_BUREAU', 'credit_status', 'DAYS_CREDIT_UPDATE', 
                              'CREDIT_DAY_OVERDUE', 'count_month_dpd', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'AMT_CREDIT_SUM_DEBT']
    list_feature_name_nominal = ['is_bereau_history']
    list_feature_name_numeric = ['cnt_active_loan', 'last_days_update', 'cnt_active_loan_past_due', 'cnt_closed_loan_past_due', 
                                 'total_credit_day_overdue', 'max_days_credit_enddate', 'min_days_enddate_fact', 'total_amt_credit_sum_debt']

    df_bereau_selected = df_bereau_prep.loc[:, list_bereau_col_select]
    df_bereau_history = df_id_ref.merge(df_bereau_selected, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')

    # column generate
    df_bereau_history['is_bereau_history'] = df_bereau_history['SK_ID_BUREAU'].notna()
    df_bereau_history['is_past_due'] = df_bereau_history[['CREDIT_DAY_OVERDUE', 'count_month_dpd']].max(axis=1).gt(0)

    filter_loan_active = df_bereau_history['credit_status'].eq('Active')
    filter_loan_closed = df_bereau_history['credit_status'].eq('Closed')

    # feature engineering
    arr_feature_is_bereau_history = df_bereau_history.groupby('SK_ID_CURR')['is_bereau_history'].any()
    df_feature_is_bereau_history = arr_feature_is_bereau_history.reset_index()

    arr_feature_num_active_loan = df_bereau_history.loc[filter_loan_active].groupby('SK_ID_CURR').size()
    df_feature_n_active_loan = arr_feature_num_active_loan.rename('cnt_active_loan').reset_index()

    arr_feature_latest_application_day = df_bereau_history.groupby('SK_ID_CURR')['DAYS_CREDIT_UPDATE'].min()
    df_feature_latest_application_day = arr_feature_latest_application_day.rename('last_days_update').reset_index()

    arr_feature_cnt_active_loan_past_due = df_bereau_history.loc[filter_loan_active].groupby('SK_ID_CURR')['is_past_due'].sum()
    df_feature_cnt_active_loan_past_due = arr_feature_cnt_active_loan_past_due.rename('cnt_active_loan_past_due').reset_index()

    arr_feature_cnt_closed_loan_past_due = df_bereau_history.loc[filter_loan_closed].groupby('SK_ID_CURR')['is_past_due'].sum()
    df_feature_cnt_closed_loan_past_due = arr_feature_cnt_closed_loan_past_due.rename('cnt_closed_loan_past_due').reset_index()

    arr_feature_total_credit_day_overdue = df_bereau_history.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].sum()
    df_feature_total_credit_day_overdue = arr_feature_total_credit_day_overdue.rename('total_credit_day_overdue').reset_index()

    arr_feature_max_remaining_bereau_date = df_bereau_history.groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].max()
    df_feature_max_remaining_bereau_date = arr_feature_max_remaining_bereau_date.rename('max_days_credit_enddate').reset_index()
    
    arr_feature_min_enddate_closed_loan = df_bereau_history.loc[filter_loan_closed].groupby('SK_ID_CURR')['DAYS_ENDDATE_FACT'].min()
    df_feature_min_enddate_closed_loan = arr_feature_min_enddate_closed_loan.rename('min_days_enddate_fact').reset_index()

    arr_feature_total_credit_debt = df_bereau_history.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].sum()
    df_feature_total_credit_debt = arr_feature_total_credit_debt.rename('total_amt_credit_sum_debt').reset_index()

    # merge all features together
    list_dfs = [df_id_ref, df_feature_is_bereau_history, df_feature_n_active_loan, df_feature_latest_application_day, df_feature_cnt_active_loan_past_due,
                df_feature_cnt_closed_loan_past_due, df_feature_total_credit_day_overdue, df_feature_max_remaining_bereau_date, df_feature_min_enddate_closed_loan,
                df_feature_total_credit_debt]
    df_features = reduce(lambda left, right: pd.merge(left, right, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left'), list_dfs)

    # preprocess feature value
    df_features['cnt_active_loan'] = df_features['cnt_active_loan'].fillna(0)
    df_features.loc[~df_features['is_bereau_history'], 'total_credit_day_overdue'] = np.nan
    df_features.loc[~df_features['is_bereau_history'], 'total_amt_credit_sum_debt'] = np.nan

    # update feature names
    if list_col_name_nominal is not None:
        list_col_name_nominal.extend(list_feature_name_nominal)
    if list_col_name_numeric is not None:
        list_col_name_numeric.extend(list_feature_name_numeric)
    
    return df_features


def preprocess_df_prev_application_data(df_prev_app: pd.DataFrame) -> pd.DataFrame:
    # filter df scope
    list_category_contract_type = ('Consumer loans', 'Cash loans', 'Revolving loans')
    filter_name_contract_type = df_prev_app['NAME_CONTRACT_TYPE'].isin(list_category_contract_type)
    filter_flag_last_appl = df_prev_app['FLAG_LAST_APPL_PER_CONTRACT'].eq('Y')

    df_prev_app = df_prev_app.loc[filter_name_contract_type & filter_flag_last_appl]

    # fixing date error
    for col in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
        filter_weird_days = df_prev_app[col].gt(0)
        df_prev_app.loc[filter_weird_days, col] = np.nan

    return df_prev_app


def merge_prev_app_features(df_id_ref: pd.DataFrame, df_prev_app_prep: pd.DataFrame, 
                            list_col_name_numeric: Optional[List[str]] = None, list_col_name_nominal: Optional[List[str]] = None) -> pd.DataFrame:
    list_feature_name_nominal = ['is_prev_app_history', 'is_all_loan_approved', 'is_loan_refused_within_curr_period', 
                                 'is_loan_canceled_within_curr_period', 'is_req_ins_on_appr']
    list_feature_name_numeric = ['latest_loan_approval', 'ratio_last_appl_to_last_appr', 'max_amt_credit', 'min_days_last_due']

    df_prev_app_history = df_id_ref.merge(df_prev_app_prep, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')

    df_1 = df_prev_app_history.groupby('SK_ID_CURR')['SK_ID_PREV'].max().notna().reset_index().rename(columns={'SK_ID_PREV': 'is_prev_app_history'})
    
    filter_status_approved = df_prev_app_history['NAME_CONTRACT_STATUS'].eq('Approved')
    filter_status_refused = df_prev_app_history['NAME_CONTRACT_STATUS'].eq('Refused')
    filter_status_canceled = df_prev_app_history['NAME_CONTRACT_STATUS'].eq('Canceled')

    df_n_prev_app = df_prev_app_history.groupby('SK_ID_CURR').size().rename('cnt_prev_appl').reset_index()
    df_n_appr_prev_app = df_prev_app_history.loc[filter_status_approved].groupby('SK_ID_CURR').size().rename('cnt_appr_prev_appl').reset_index()
    df_2 = df_n_prev_app.merge(df_n_appr_prev_app, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
    df_2['cnt_appr_prev_appl'] = df_2['cnt_appr_prev_appl'].fillna(0)
    df_2['is_all_loan_approved'] = df_2['cnt_prev_appl'].eq(df_2['cnt_appr_prev_appl'])
    df_2.drop(columns=['cnt_appr_prev_appl', 'cnt_prev_appl'], inplace=True)

    day_cutoff = -270
    filter_day_cutoff = df_prev_app_history['DAYS_DECISION'].gt(day_cutoff)
    
    df_id_refuse = df_prev_app_history.loc[filter_status_refused & filter_day_cutoff, ['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates(subset='SK_ID_CURR')
    df_3 = df_id_ref.merge(df_id_refuse, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
    df_3['is_loan_refused_within_curr_period'] = df_3['SK_ID_PREV'].notna()
    df_3.drop(columns='SK_ID_PREV', inplace=True)

    df_id_canceled = df_prev_app_history.loc[filter_status_canceled & filter_day_cutoff, ['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates(subset='SK_ID_CURR')
    df_4 = df_id_ref.merge(df_id_canceled, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
    df_4['is_loan_canceled_within_curr_period'] = df_4['SK_ID_PREV'].notna()
    df_4.drop(columns='SK_ID_PREV', inplace=True)

    df_latest_info = df_prev_app_history.groupby('SK_ID_CURR')['DAYS_DECISION'].max().reset_index().rename(columns={'DAYS_DECISION': 'latest_information_day'})
    df_latest_loan_approval = df_prev_app_history.loc[filter_status_approved].groupby('SK_ID_CURR')['DAYS_DECISION'].max().reset_index().rename(columns={'DAYS_DECISION': 'latest_loan_approval'})
    df_5 = df_latest_info.merge(df_latest_loan_approval, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
    df_5['ratio_last_appl_to_last_appr'] = df_5['latest_information_day'] / df_5['latest_loan_approval']
    df_5.drop(columns='latest_information_day', inplace=True)

    df_6 = df_prev_app_history.groupby('SK_ID_CURR')['AMT_CREDIT'].max().reset_index().rename(columns={'AMT_CREDIT': 'max_amt_credit'})

    df_7 = df_prev_app_history.groupby('SK_ID_CURR')['DAYS_LAST_DUE'].min().reset_index().rename(columns={'DAYS_LAST_DUE': 'min_days_last_due'})

    df_8 = df_prev_app_history.groupby('SK_ID_CURR')['NFLAG_INSURED_ON_APPROVAL'].any().reset_index().rename(columns={'NFLAG_INSURED_ON_APPROVAL': 'is_req_ins_on_appr'})
    
    list_dfs = [df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8]
    df_features = reduce(lambda left, right: pd.merge(left, right, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner'), list_dfs)

    # update feature names
    if list_col_name_nominal is not None:
        list_col_name_nominal.extend(list_feature_name_nominal)
    if list_col_name_numeric is not None:
        list_col_name_numeric.extend(list_feature_name_numeric)
    
    return df_features


def merge_pos_cash_bal(df_id_ref: pd.DataFrame, df_pos_cash_bal: pd.DataFrame, n_years: int = 6,
                       list_col_name_numeric: Optional[List[str]] = None, list_col_name_nominal: Optional[List[str]] = None) -> pd.DataFrame:
    list_feature_name_nominal = []
    list_feature_name_numeric = ['cnt_mth_past_due']

    n_months_minimum = -12 * n_years

    df_pos_cash_bal['is_day_past_due'] = df_pos_cash_bal['SK_DPD'].gt(0)
    df_prev_loan_dpd_cnt = df_pos_cash_bal.loc[df_pos_cash_bal['MONTHS_BALANCE'].gt(n_months_minimum)].groupby(['SK_ID_CURR'])['is_day_past_due'].sum()
    df_feature = df_prev_loan_dpd_cnt.reset_index().rename(columns={'is_day_past_due': 'cnt_mth_past_due'})
    df_feature_ref = df_id_ref.merge(df_feature, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')

    # update feature names
    if list_col_name_nominal is not None:
        list_col_name_nominal.extend(list_feature_name_nominal)
    if list_col_name_numeric is not None:
        list_col_name_numeric.extend(list_feature_name_numeric)

    return df_feature_ref


def merge_installment_payment(df_id_ref: pd.DataFrame, df_inst_paymnt: pd.DataFrame,
                              list_col_name_nominal: Optional[List[str]] = None, list_col_name_numeric: Optional[List[str]] = None) -> pd.DataFrame:
    list_feature_name_nominal = []
    list_feature_name_numeric = ['mean_day_of_payment_bf_deadline', 'ratio_payment_short']

    tol = 1e-3

    # drop na
    df_inst_paymnt = df_inst_paymnt.dropna().copy()

    # clean weird payment amount
    arr_diff_payment = df_inst_paymnt['AMT_PAYMENT'] - df_inst_paymnt['AMT_INSTALMENT']
    filter_exceed_payment = arr_diff_payment.gt(tol)
    df_inst_paymnt['AMT_PAYMENT'] = np.where(filter_exceed_payment, df_inst_paymnt['AMT_INSTALMENT'], df_inst_paymnt['AMT_PAYMENT'])

    # clearn weird payment date
    df_inst_paymnt['days_diff'] = df_inst_paymnt['DAYS_ENTRY_PAYMENT'] - df_inst_paymnt['DAYS_INSTALMENT']
    
    filter_payment_before_31 = df_inst_paymnt['days_diff'].lt(-31)
    filter_payment_after_31 = df_inst_paymnt['days_diff'].gt(31)

    df_inst_paymnt.loc[filter_payment_before_31, 'days_diff'] = -31
    df_inst_paymnt.loc[filter_payment_after_31, 'days_diff'] = 31

    df_1 = df_inst_paymnt.groupby('SK_ID_CURR')['days_diff'].mean().reset_index().rename(columns={'days_diff': 'mean_day_of_payment_bf_deadline'})
    
    df_inst_paymnt['amt_payment_shortage'] = df_inst_paymnt['AMT_PAYMENT'] - df_inst_paymnt['AMT_INSTALMENT']
    df_inst_paymnt['is_amt_payment_short'] = df_inst_paymnt['amt_payment_shortage'] < -tol
    arr_cnt_payment_short = df_inst_paymnt.groupby('SK_ID_CURR')['is_amt_payment_short'].sum()
    arr_cnt_payment = df_inst_paymnt.groupby('SK_ID_CURR').size()
    arr_ratio_payment_shrt = arr_cnt_payment_short / arr_cnt_payment
    df_2 = arr_ratio_payment_shrt.rename('ratio_payment_short').reset_index()

    df_feature = df_id_ref.merge(df_1, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')
    df_feature = df_feature.merge(df_2, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='left')

    # update feature names
    if list_col_name_nominal is not None:
        list_col_name_nominal.extend(list_feature_name_nominal)
    if list_col_name_numeric is not None:
        list_col_name_numeric.extend(list_feature_name_numeric)

    return df_feature


def cross_ref_credit_amt_features(df: pd.DataFrame, list_col_name_nominal: Optional[List[str]], 
                                  list_col_name_numeric: Optional[List[str]]) -> pd.DataFrame:
    list_feature_name_nominal = []
    list_feature_name_numeric = ['ratio_max_amt_credit']

    arr_max_amt_credit_ratio = df['AMT_CREDIT'] / df['max_amt_credit']
    df['ratio_max_amt_credit'] = np.where(df['max_amt_credit'].notna(), arr_max_amt_credit_ratio, np.nan)
    df['ratio_max_amt_credit'] = df['ratio_max_amt_credit'].replace(np.inf, np.nan)

    # update feature names
    if list_col_name_nominal is not None:
        list_col_name_nominal.extend(list_feature_name_nominal)
    if list_col_name_numeric is not None:
        list_col_name_numeric.extend(list_feature_name_numeric)

    return df


def clean_dummy_col_name(list_dummy_col: List[str]) -> List[str]:
    def clean_name(x: str) -> str:
        x = x.replace(',', '')
        x = x.replace('/', '')
        x = x.replace(':', '')
        x = x.replace('-', '_')
        x = x.replace(' ', '_')
        x = x.replace('__', '_')
        return x
    
    return list(map(clean_name, list_dummy_col))
