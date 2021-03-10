from typing import Dict, List
from tensorflow.keras import backend as K
import numpy as np


def calc_metrics(gt: np.ndarray, pred: np.ndarray, metrics: List[str]) -> Dict[str, List[float]]:
    assert gt.shape == pred.shape, f'Shapes of gt and pred must be equal. gt: {gt.shape}, pred: {pred.shape}'
    metric_mappings = {
        'iou': iou_all,
        'iou_strict': iou_all_strict,
        'acc': acc,
    }
    return {metric_name: metric_mappings[metric_name](gt, pred) for metric_name in metrics}


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


def to_strict(pred):
    n_cl = pred.shape[-1]
    c = np.argmax(pred, axis=-1)
    return np.eye(n_cl)[c] # same as to_categorical


def iou_per_class(y_true, y_pred):
    iou_vals = []
    n_cl = y_pred.shape[-1]
    for i in range(n_cl):
        iou_vals.append(iou(y_true[...,i], y_pred[...,i]))
    return iou_vals


def iou_all(y_true, y_pred):
    return iou_per_class(y_true, y_pred) + [iou(y_true, y_pred)]


def iou_all_strict(y_true, y_pred):
    y_pred_strict = to_strict(y_pred)
    return iou_per_class(y_true, y_pred_strict) + [iou(y_true, y_pred_strict)]


def acc(y_true: np.ndarray, y_pred: np.ndarray):
    n_cl = y_pred.shape[-1]
    y_pred_a = np.argmax(y_pred, axis=-1)
    y_true_a = np.argmax(y_true, axis=-1)
    acc_v = np.sum(y_pred_a == y_true_a) / y_pred_a.size
    return [acc_v] * (n_cl + 1)