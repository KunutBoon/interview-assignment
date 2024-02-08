import os
import re
import json
import math
import logging
from typing import Optional, List, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

# get project root dir
DIR_PROJECT_ROOT = os.path.dirname(os.getcwd())

# get all project sub dirs
DIR_CONFIG = os.path.join(DIR_PROJECT_ROOT, 'config')
DIR_DATA = os.path.join(DIR_PROJECT_ROOT, 'data')

DIR_RAW_DATA = os.path.join(DIR_DATA, 'raw')
DIR_PREPROCESS_DATA_PROD = os.path.join(DIR_DATA, 'production')

DIR_PREPROCESS_DATA_ARTIFACTS = os.path.join(DIR_PREPROCESS_DATA_PROD, 'artifacts')
DIR_PREPROCESS_DATA_MODELING = os.path.join(DIR_PREPROCESS_DATA_PROD, 'process_data')

DIR_PREPROCESS_DATA_MODELING_TRAIN = os.path.join(DIR_PREPROCESS_DATA_MODELING, 'train')
DIR_PREPROCESS_DATA_MODELING_TEST = os.path.join(DIR_PREPROCESS_DATA_MODELING, 'test')
DIR_PREPROCESS_DATA_INFERENCE = os.path.join(DIR_PREPROCESS_DATA_MODELING, 'inference')

DIR_INFERENCE_OUTPUT = os.path.join(DIR_PREPROCESS_DATA_PROD, 'inference_output')

DIR_MODEL = os.path.join(DIR_PROJECT_ROOT, 'model_pipeline')


def get_resource_utilize() -> int:
    cpu_count = os.cpu_count()
    if cpu_count > 1:
        half_cpu_cnt = cpu_count / 2
        resource_util = math.floor(half_cpu_cnt)
    else:
        resource_util = cpu_count

    return resource_util


def cre8_dir(dir: str, logger: Optional[logging.RootLogger] = None) -> None:
    if not os.path.exists(dir):
        os.makedirs(dir)

        if logger is not None:
            logger.debug("Directory created at {}".format(dir))


def cre8_file_suffix() -> str:
    datetime_now = datetime.today()
    datetime_suffix = datetime_now.strftime(r'%Y_%m_%d_%H_%M_%S')

    return datetime_suffix


def get_latest_file_dir(file_dir: str, file_type: str = '.csv', file_prefix: Optional[str] = None) -> str:
    if file_prefix is None:
        file_prefix = ''
    
    str_file_type_norm = file_type.replace('.', '')
    str_file_pat = r"^{prefix}.+\.{type}$".format(prefix=file_prefix, type=str_file_type_norm)
    file_pat = re.compile(str_file_pat)

    list_files = os.listdir(file_dir)
    list_files_match = [f_name for f_name in list_files if file_pat.match(f_name)]

    if len(list_files_match) == 0:
        raise FileNotFoundError(
            "There is no files with specified file type \"{}\" in directory: {}".format(file_type, file_dir)
        )
    else:
        list_files_match_sorted = sorted(list_files_match)
        return list_files_match_sorted[-1]


def get_logger_level(log_level: Optional[str]) -> int:
    _log_levels = ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')

    if log_level is None:
        return logging.NOTSET
    else:
        if log_level not in _log_levels:
            raise ValueError(
                "Expect log levels [\"{list_log_levels}\"], got \"{log_level}\"".format(
                    list_log_levels="\", \"".join(_log_levels), 
                    log_level=log_level
                )
            )
        else:
            if log_level == 'NOTSET':
                return logging.NOTSET
            elif log_level == 'DEBUG':
                return logging.DEBUG
            elif log_level == 'INFO':
                return logging.INFO
            elif log_level == 'WARNING':
                return logging.WARNING
            elif log_level == 'ERROR':
                return logging.ERROR
            else:
                return logging.CRITICAL


def get_logger(logger_name: str, path_log_filename: Optional[str] = None, log_level: str = 'DEBUG') -> logging.RootLogger:
    # create main logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.DEBUG)
    logger.propagate = False  # prevent duplicate log msg

    # logging format
    logging_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    log_file_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(funcName)s - %(levelname)s : %(message)s in %(pathname)s:%(lineno)d",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # get level
    level = get_logger_level(log_level)

    # create console handler for logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=level)
    console_handler.setFormatter(logging_format)

    # create file handler for logger
    if path_log_filename is not None:
        dir_log_filename = os.path.dirname(path_log_filename)
        cre8_dir(dir_log_filename)

        file_handler = logging.FileHandler(path_log_filename)
        file_handler.setLevel(level=logging.DEBUG)
        file_handler.setFormatter(log_file_format)
        # add handler
        logger.addHandler(file_handler)

    # add handler
    logger.addHandler(console_handler)

    return logger


def save_csv_file(df: pd.DataFrame, save_path: str, logger: logging.RootLogger) -> None:
    df.to_csv(save_path, index=True)
    logger.debug("Csv file saved at {}".format(save_path))


def load_csv_file(load_path: str, logger: logging.RootLogger) -> pd.DataFrame:
    df = pd.read_csv(load_path, index_col=0)
    logger.debug("Csv file loaded from {}".format(load_path))
    return df


def save_list(list_: List, save_path: str, logger: logging.RootLogger) -> None:
    joblib.dump(list_, save_path)
    logger.debug("List file saved at {}".format(save_path))


def load_list(load_path: str, logger: logging.RootLogger) -> List:
    list_ = joblib.load(load_path)
    logger.debug("List file loaded from {}".format(load_path))
    return list_


def save_dict(dict_: Dict[str, List], save_path: str, logger: logging.RootLogger) -> None:
    with open(save_path, 'w') as f:
        json.dump(dict_, f)
    logger.debug("Dict file saved at {}".format(save_path))


def load_dict(load_path: str, logger: logging.RootLogger) -> Dict:
    with open(load_path) as f:
        dict_ = json.load(f)
    logger.debug("Dict file loaded from {}".format(load_path))
    return dict_


def save_ndarray(arr: np.array, save_path: str, logger: logging.RootLogger) -> None:
    with open(save_path, 'wb') as f:
        np.save(f, arr)
    logger.debug("Numpy array file saved at {}".format(save_path))


def load_ndarray(load_path: str, logger: logging.RootLogger) -> np.array:
    with open(load_path, 'rb') as f:
        arr = np.load(f)
    logger.debug("Numpy array file loaded from {}".format(load_path))
    return arr


def save_model(model, save_path: str, logger: logging.RootLogger) -> None:
    joblib.dump(model, save_path)
    logger.debug("Sklearn model obj saved at {}".format(save_path))


def load_model(load_path: str, logger: logging.RootLogger):
    model = joblib.load(load_path)
    logger.debug("Sklearn model obj loaded from {}".format(load_path))
    return model
