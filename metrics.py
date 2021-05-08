from collections import namedtuple
from typing import List

import numpy as np
from tensorflow.keras import backend as K


exIoU = namedtuple('IoU', ['iou', 'intersection', 'union'])
exAcc = namedtuple('Acc', ['acc', 'correct', 'total'])


def iou_tf(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def iou(y_true, y_pred, smooth=1.0) -> exIoU:
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return exIoU((intersection + smooth) / (union + smooth), intersection, union)


def to_strict(pred: np.ndarray) -> np.ndarray:
    n_cl = pred.shape[-1]
    c = np.argmax(pred, axis=-1)
    return np.eye(n_cl)[c] # same as to_categorical


def iou_per_class(y_true, y_pred, smooth=1.0) -> List[exIoU]:
    iou_vals = []
    n_cl = y_pred.shape[-1]
    for i in range(n_cl):
        iou_vals.append(iou(y_true[...,i], y_pred[...,i], smooth))
    return iou_vals


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> exAcc:
    y_pred_a = np.argmax(y_pred, axis=-1)
    y_true_a = np.argmax(y_true, axis=-1)
    correct = np.sum(y_pred_a == y_true_a)
    return exAcc(correct / y_pred_a.size, correct=correct, total=y_pred_a.size)


def joint_iou(ious: List[exIoU], smooth=1.0) -> exIoU:
    intersection = sum(i.intersection for i in ious)
    union = sum(i.union for i in ious)
    return exIoU((intersection + smooth) / (union + smooth), intersection, union)


def joint_acc(accs: List[exAcc]) -> exAcc:
    correct = sum(a.correct for a in accs)
    total = sum(a.total for a in accs)
    return exAcc(correct / total, correct, total)

