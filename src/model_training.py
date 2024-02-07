import os
import math
import logging
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb

from utils import DIR_PROJECT_ROOT, load_dict, get_resource_utilize

N_JOBS = get_resource_utilize()
RANDOM_SEED = 99

path_training_cfg = os.path.join(DIR_PROJECT_ROOT, 'config', 'training', 'hyper_param_config.json')


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features: int):
        self.n_features = n_features

    def fit(self, X: np.array, y: np.array):
        filter_out_row_nan = np.isnan(X).any(axis=1)

        y_filter = y[~filter_out_row_nan]
        X_filter = X[~filter_out_row_nan]

        dict_class_weight = get_dict_class_weight(y_filter)

        model_feature_selection = _get_model(model_type='rf')
        model_feature_selection.set_params(max_depth=None, n_jobs=N_JOBS, class_weight=dict_class_weight)
        feature_selector = SelectFromModel(estimator=model_feature_selection, threshold=-np.inf, max_features=self.n_features)
        feature_selector.fit(X_filter, y_filter)

        arr_bool_selected = feature_selector.get_support()
        self.get_support = arr_bool_selected

        return self
    
    def transform(self, X):
        X_selected = X[:, self.get_support]

        return X_selected


def _parse_training_cfg(logger: logging.RootLogger) -> Tuple[int, str, Dict]:
    dict_hyper_params = load_dict(path_training_cfg, logger=logger)

    # parse dict
    n_features = dict_hyper_params['n_features']
    model_type = dict_hyper_params['model']
    model_hyperparams = dict_hyper_params['hyperparameters']

    return n_features, model_type, model_hyperparams


def _get_model(model_type: str) -> Union[DecisionTreeClassifier, RandomForestClassifier, xgb.XGBClassifier]:
    if model_type not in ('xgb', 'rf', 'dt'):
        raise ValueError(
            "Incorrect model type"
        )
    else:
        if model_type == 'xgb':
            return xgb.XGBClassifier()
        elif model_type == 'dt':
            return DecisionTreeClassifier()
        else:
            return RandomForestClassifier()


def preprocess_nominal(df: pd.DataFrame, list_nominal_col: List[str], return_objects_for_encoder: bool = False, 
                       dict_category: Optional[Dict] = None) -> Union[Tuple[pd.DataFrame, List[str], List[List[str]]], pd.DataFrame]:
    df_nominal = df.loc[:, list_nominal_col]

    # preprocess on bool columns
    list_col_bool = df_nominal.select_dtypes(include='bool').columns.tolist()
    df.loc[:, list_col_bool] = df[list_col_bool].astype(int)

    # preprocess on obj columns
    if return_objects_for_encoder:
        list_categories_ = list()
        list_col_object = df_nominal.select_dtypes(include='object').columns.tolist()
        for col in list_col_object:
            list_selected_category = dict_category[col]
            list_unique_category = df_nominal[col].unique().tolist()
            if len(list_selected_category) == len(list_unique_category):
                list_selected_category.pop(0)
            list_categories_.append(list_selected_category)

        return df, list_col_object, list_categories_
    else:
        return df
    

def get_dict_class_weight(y: pd.Series) -> Dict[Union[int, float], float]:
    arr_unique_class = np.unique(y)
    arr_class_weight = compute_class_weight(class_weight="balanced", classes=arr_unique_class, y=y)
    dict_class_weight = dict(zip(arr_unique_class, arr_class_weight))

    return dict_class_weight
    

def cre8_sklearn_pipeline(y: pd.Series, list_col_num: List[str], list_col_cat: List[str], list_categories_: List[List], 
                          logger: logging.getLogger):
    # parse cfg
    n_features, model_type, model_hyperparams = _parse_training_cfg(logger)
    model_training = _get_model(model_type=model_type)
    if model_type != 'xgb':
        dict_class_weight = get_dict_class_weight(y)
        model_training.set_params(class_weight=dict_class_weight)

    # processing numeric
    numeric_processor = Pipeline(
        steps=[('scaler', MinMaxScaler())]
    )
    # processing category
    categorical_processor = Pipeline(
        steps=[('encoder', OneHotEncoder(categories=list_categories_, handle_unknown='ignore', sparse_output=False))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_processor, list_col_num), 
            ('category', categorical_processor, list_col_cat)
        ]
    )

    # feature selector
    feature_selector_obj = FeatureSelector(n_features=n_features)

    # model
    model_training.set_params(**model_hyperparams)

    # pipeline
    model_pipeline = Pipeline(
        steps=[
            ('processor', preprocessor),
            ('selector', feature_selector_obj),
            ('model', model_training)
        ]
    )

    return model_pipeline
