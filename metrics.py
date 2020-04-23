import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from tensorflow_core.python.layers.core import flatten
from tensorflow_core import reduce_sum

def calc_metrics(mask_gt: np.ndarray, mask_pred: np.ndarray, metrics: list, num_classes: int):

    assert mask_gt.shape == mask_pred.shape, 'Shapes of gt and pred must be equal in calc_metrics'

    metrics_dict = {'iou' : iou_multiclass}
    results = []

    for metric in metrics:
        results.append((metric, metrics_dict[metric](mask_gt, mask_pred, num_classes)))

    return results

def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def iou_binary(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)

def iou_multiclass(y_true, y_pred, num_classes):
    results = []
    for i in range(num_classes):
        results.append(iou_binary(y_true[...,i], y_pred[...,i]))
    return results

def dice(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.cast(flatten(y_true), tf.float32)
    y_pred_f = tf.cast(flatten(y_pred), tf.float32)
    intersection = reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (reduce_sum(y_true_f) + reduce_sum(y_pred_f) + smooth)

def weighted_dice(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * reduce_sum(w * intersection) + smooth) / (reduce_sum(w * m1) + reduce_sum(w * m2) + smooth)
    return score