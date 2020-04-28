from tensorflow_core.python.keras.losses import binary_crossentropy
from metrics import dice, weighted_dice

def combined_loss(alpha):
    def wrapper(loss1, loss2):
        return lambda x, y: alpha * loss1(x, y) + (1 - alpha) * loss2(x, y)
    return wrapper

def dice_loss(y_true, y_pred):
    return 1 - dice(y_true, y_pred)

def weighted_dice_loss(y_true, y_pred, weights):
    return 1 - weighted_dice(y_true, y_pred, weights)

def bce_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)

def bce_dice_loss(y_true, y_pred, alpha=1.0, beta=1.0):
    return binary_crossentropy(y_true, y_pred) * alpha + dice_loss(y_true, y_pred) * beta

# def weighted_dice_loss(y_true, y_pred):
#     y_true = K.cast(y_true, 'float32')
#     y_pred = K.cast(y_pred, 'float32')
#     # if we want to get same size of output, kernel size must be odd number
#     if K.int_shape(y_pred)[1] == 128:
#         kernel_size = 11
#     elif K.int_shape(y_pred)[1] == 256:
#         kernel_size = 21
#     elif K.int_shape(y_pred)[1] == 512:
#         kernel_size = 21
#     elif K.int_shape(y_pred)[1] == 1024:
#         kernel_size = 41
#     else:
#         raise ValueError('Unexpected image size')
#     averaged_mask = K.pool2d(
#         y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
#     border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
#     weight = K.ones_like(averaged_mask)
#     w0 = K.sum(weight)
#     weight += border * 2
#     w1 = K.sum(weight)
#     weight *= (w0 / w1)
#     loss = 1 - weighted_dice_coeff(y_true, y_pred, weight)
#     return loss