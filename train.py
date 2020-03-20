import tensorflow as tf
from data_utils import get_imgs_masks
from tensorflow.keras.utils import to_categorical
from unet import custom_unet
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from metrics import iou, iou_multiclass
from utils import plot_segm_history
import os
import numpy as np
from callbacks import TestResults
from generators import PatchGenerator
from time import time

def train(n_classes, n_layers, n_filters, path, epochs, batch_size, patch_size, show_history=True):
    
    t1 = time()
    x_train, x_val, y_train, y_val = get_imgs_masks(path)
    t2 = time()
    print(f'load time: {t2-t1} seconds')
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    print('Train data size: {} images and {} masks'.format(x_train.shape[0], y_train.shape[0]))
    print('Validation data size: {} images and {} masks'.format(x_val.shape[0], y_val.shape[0]))

    aug_factor = 5
    steps_per_epoch = np.ceil((x_train.shape[0] * x_train.shape[1] * x_train.shape[2] * aug_factor) / (batch_size * patch_size * patch_size)).astype('int')
    print('Steps per epoch: {}'.format(steps_per_epoch))

    y_train = to_categorical(y_train, num_classes=n_classes)
    y_val = to_categorical(y_val, num_classes=n_classes)

    input_shape = (patch_size, patch_size, 3)

    model = custom_unet(
        input_shape,
        n_classes=n_classes,
        filters=n_filters,
        use_batch_norm=True,
        n_layers=n_layers,
        output_activation='softmax'
    )

    model_filename = 'model_v1.h5'
    callback_checkpoint = ModelCheckpoint(
        model_filename, 
        verbose=1, 
        monitor='val_loss', 
        save_best_only=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    model.compile(
        optimizer=Adam(), 
        loss = 'categorical_crossentropy',
        metrics=[iou]
    )

    callback_test = TestResults(
        images=x_val, 
        masks=y_val, 
        model=model, 
        n_classes=n_classes,
        batch_size=batch_size,
        patch_size=patch_size,
        offset=2 * n_layers,
        output_path='output',
        all_metrics=['iou']
    )

    csv_logger = CSVLogger('training.log')

    train_generator = PatchGenerator(images=x_train, masks=y_train, patch_size=patch_size, batch_size=batch_size, augment=True)
    steps_per_epoch = 10
    history = model.fit(
        iter(train_generator),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[callback_checkpoint, callback_test, csv_logger, reduce_lr],
    )

    if show_history:
        plot_segm_history(history, 'output')

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "input", "dataset")
    train(n_classes=4, n_layers=3, n_filters=4, epochs=2, path=path, batch_size=8, patch_size=512)