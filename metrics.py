#import plaidml.keras
#plaidml.keras.install_backend()
#from keras import backend as K
import numpy as np

def calc_metrics(mask_gt : np.ndarray, mask_pred : np.ndarray, metrics : list):

    metrics_dict = {'iou' : iou}
    results = []

    for metric in metrics:
        results.append(metrics_dict[metric](mask_gt, mask_pred))

    return results

# def iou(y_true, y_pred, smooth=1.):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def iou(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)