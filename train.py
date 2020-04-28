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
import functools
import losses
import shutil
from config import classes_mask, train_params, classes_weights
from sklearn.utils import class_weight

def train(n_classes, n_layers, n_filters, path, epochs, batch_size, patch_size, overlay, output_path, show_history=True):
    
    print('Loading images and masks..')
    t1 = time()
    x_train, x_test, y_train, y_test, train_names, test_names = get_imgs_masks(path, True, True)
    t2 = time()
    print(f'Images and masks load time: {t2-t1} seconds')
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    print('Train data size: {} images and {} masks'.format(x_train.shape[0], y_train.shape[0]))
    print('Test data size: {} images and {} masks'.format(x_test.shape[0], y_test.shape[0]))

    aug_factor = train_params["aug_factor"]
    steps_per_epoch = np.ceil((x_train.shape[0] * x_train.shape[1] * x_train.shape[2] * aug_factor) / (batch_size * patch_size * patch_size)).astype('int')
    print('Steps per epoch: {}'.format(steps_per_epoch))

    # print('Computing weights..')
    # t1 = time()
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train.flatten())
    # t2 = time()
    # class_weights = dict(enumerate(class_weights))
    # print(f'Weights computed in {t2 - t1} seconds')
    print(f'Weights: {classes_weights}')

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

    # custom_loss = functools.partial(losses.weighted_dice_loss, weights=[classes_weights[i] for i in classes_weights.keys()])
    custom_loss = functools.partial(losses.dice_loss)
    custom_loss.__name__ = 'custom_loss'

    model.compile(
        optimizer=Adam(), 
        loss = custom_loss,
        metrics=[iou]
    )

    callback_test = TestResults(
        images=x_test, 
        masks=y_test,
        names=test_names, 
        model=model, 
        n_classes=n_classes,
        batch_size=batch_size,
        patch_size=patch_size,
        overlay=overlay,
        offset=2 * n_layers,
        output_path=output_path,
        all_metrics=['iou']
    )

    early_stop = EarlyStopping(
        monitor='loss',
        min_delta=0.001,
        patience=7,
        restore_best_weights=True
    )
    steps_per_epoch = 5
    csv_logger = CSVLogger('training.log')
    train_generator = PatchGenerator(images=x_train, masks=y_train, names=train_names, patch_size=patch_size, batch_size=batch_size, full_augment=train_params["full_augment"])
    history = model.fit(
        iter(train_generator),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[callback_checkpoint, callback_test, csv_logger, reduce_lr, early_stop]
    )

    for key in history.history:
        print(key)

    if show_history:
        plot_segm_history(history, output_path)

if __name__ == "__main__":
    if os.path.exists(train_params["output_path"]):
        shutil.rmtree(train_params["output_path"])

    train(n_classes=len(classes_mask.keys()), n_layers=train_params["n_layers"], 
            n_filters=train_params["n_filters"], epochs=train_params["epochs"], 
            path=train_params["dataset_path"], batch_size=train_params["batch_size"],
            patch_size=train_params["patch_size"], overlay=train_params["overlay"],
            output_path=train_params["output_path"])