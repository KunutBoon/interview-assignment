from typing import Union, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def accuracy_report(y_true: Union[np.array, pd.Series], y_pred: np.array) -> Dict:
    str_confusion_matrix = str(confusion_matrix(y_true, y_pred))\
                                .replace('[[', ' ' * 5 + '[[')\
                                .replace('\n ', '\n' + ' ' * 6)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return dict(
        confusion_matrix=str_confusion_matrix,
        accuracy_score=accuracy,
        precision_score=precision,
        recall_score=recall,
        f1_score=f1
    )
