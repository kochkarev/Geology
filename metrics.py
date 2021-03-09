from typing import Dict, List
from tensorflow.keras import backend as K
import numpy as np

def calc_metrics(gt: np.ndarray, pred: np.ndarray, metrics: List[str], n_classes: int) -> Dict[str, List[float]]:
    assert gt.shape == pred.shape, f'Shapes of gt and pred must be equal. gt: {gt.shape}, pred: {pred.shape}'
    metric_mappings = {
        'iou': iou_all,
    }
    res = dict()
    for metric_name in metrics:
        res[metric_name] = metric_mappings[metric_name](gt, pred, n_classes)
    return res

def iou_tf(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def iou(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)

def iou_per_class(y_true, y_pred, n_classes):
    iou_vals = []
    for i in range(n_classes):
        iou_vals.append(iou(y_true[...,i], y_pred[...,i]))
    return iou_vals

def iou_all(y_true, y_pred, n_classes):
    return iou_per_class(y_true, y_pred, n_classes) + [iou(y_true, y_pred)]
