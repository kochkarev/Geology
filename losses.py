from tensorflow_core.python.keras.losses import binary_crossentropy
from metrics import dice

def combined_loss(alpha):
    def wrapper(loss1, loss2):
        return lambda x, y: alpha * loss1(x, y) + (1 - alpha) * loss2(x, y)
    return wrapper

def dice_loss(y_true, y_pred):
    return 1 - dice(y_true, y_pred)

def bce_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)

def bce_dice_loss(y_true, y_pred, alpha=1.0, beta=1.0):
    return binary_crossentropy(y_true, y_pred) * alpha + dice_loss(y_true, y_pred) * beta