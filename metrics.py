#import plaidml.keras
#plaidml.keras.install_backend()
from keras import backend as K
import numpy as np

def calc_metrics(mask_gt : np.ndarray, mask_pred : np.ndarray, metrics : list, num_classes : int):

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
        results.append(iou_binary(np.array(np.where(y_true[...,i] > 0, 1, 0), dtype=np.uint8), np.array(np.where(y_pred[...,i] > 0, 1, 0), dtype=np.uint8)))
    return results