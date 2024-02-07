from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def accuracy_report(y_true: Union[np.array, pd.Series], y_pred: np.array) -> None:
    str_confusion_matrix = str(confusion_matrix(y_true, y_pred))\
                                .replace('[[', ' ' * 5 + '[[')\
                                .replace('\n ', '\n' + ' ' * 6)
    
    print("Confusion Matrix : \n{}".format(str_confusion_matrix))
    print("Accuracy Score : {0:.2f}".format(accuracy_score(y_true, y_pred)))
    print("Precision Score : {0:.2f}".format(precision_score(y_true, y_pred)))
    print("Recall Score : {0:.2f}".format(recall_score(y_true, y_pred)))
    print("F1 Score : {0:.2f}".format(f1_score(y_true, y_pred)))