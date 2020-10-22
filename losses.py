from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from metrics import dice, weighted_dice

def dice_loss(y_true, y_pred):
    return 1 - dice(y_true, y_pred)

def weighted_dice_loss(y_true, y_pred, weights):
    return 1 - weighted_dice(y_true, y_pred, weights)

def cce_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)

def cce_dice_loss(y_true, y_pred, alpha=1.0, beta=1.0):
    return categorical_crossentropy(y_true, y_pred) * alpha + dice_loss(y_true, y_pred) * beta