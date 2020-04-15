import tensorflow as tf
from data_utils import get_imgs_masks
from tensorflow.keras.utils import to_categorical
from unet import custom_unet
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
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
    x_train, x_test, y_train, y_test, train_names, _ = get_imgs_masks(path, True, True)
    t2 = time()
    print(f'Images and masks load time: {t2-t1} seconds')
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    print('Train data size: {} images and {} masks'.format(x_train.shape[0], y_train.shape[0]))
    print('Test data size: {} images and {} masks'.format(x_test.shape[0], y_test.shape[0]))

    aug_factor = 5
    steps_per_epoch = np.ceil((x_train.shape[0] * x_train.shape[1] * x_train.shape[2] * aug_factor) / (batch_size * patch_size * patch_size)).astype('int')
    print('Steps per epoch: {}'.format(steps_per_epoch))

    y_train = to_categorical(y_train, num_classes=n_classes).astype(np.uint8)
    y_test = to_categorical(y_test, num_classes=n_classes).astype(np.uint8)

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
        monitor='loss', 
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
        images=x_test, 
        masks=y_test, 
        model=model, 
        n_classes=n_classes,
        batch_size=batch_size,
        patch_size=patch_size,
        offset=2 * n_layers,
        output_path='output',
        all_metrics=['iou']
    )

    early_stop = EarlyStopping(
        monitor='loss',
        min_delta=0.001,
        patience=5,
        restore_best_weights=True
    )

    csv_logger = CSVLogger('training.log')

    train_generator = PatchGenerator(images=x_train, masks=y_train, names=train_names, patch_size=patch_size, batch_size=batch_size, augment=True)
    history = model.fit(
        iter(train_generator),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[callback_checkpoint, callback_test, csv_logger, reduce_lr, early_stop],
    )

    if show_history:
        plot_segm_history(history, 'output')

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "input", "dataset")
    train(n_classes=4, n_layers=3, n_filters=16, epochs=100, path=path, batch_size=8, patch_size=512)