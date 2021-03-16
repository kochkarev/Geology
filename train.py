import tensorflow as tf
from data_utils import get_imgs_masks
from tensorflow.keras.utils import to_categorical
from unet import custom_unet
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from metrics import iou, iou_multiclass
from utils import plot_segm_history
import os
import numpy as np
from callbacks import TestResults, AdvancedCheckpoint
from generators import PatchGenerator
from time import time
import functools
import losses
import shutil
from config import classes_mask, train_params, classes_weights
from sklearn.utils import class_weight
from tensorflow.keras.models import load_model
import pickle
import argparse

def train(n_classes, n_layers, n_filters, path, epochs, batch_size, patch_size, overlay, output_path, model_name = None, show_history=True):
    
    print('Loading images and masks..')
    t1 = time()
    x_train, x_test, y_train, y_test, train_names, test_names = get_imgs_masks(os.path.join(path, "S1_v1"), load_test=True, return_names=True)
    t2 = time()
    print(f'Images and masks load time: {t2-t1} seconds')
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    print(f'Num of classes: {n_classes}')
    print(f'Classes ids: {np.unique(y_train)}')

    print('Train data size: {} images and {} masks'.format(x_train.shape[0], y_train.shape[0]))
    print('Test data size: {} images and {} masks'.format(x_test.shape[0], y_test.shape[0]))

    aug_factor = train_params["aug_factor"]
    steps_per_epoch = np.ceil((x_train.shape[0] * x_train.shape[1] * x_train.shape[2] * aug_factor) / (batch_size * patch_size * patch_size)).astype('int')
    print('Steps per epoch: {}'.format(steps_per_epoch))

    # print('Computing weights..')
    # t1 = time()
    # class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.flatten())
    # t2 = time()
    # class_weights = dict(enumerate(class_weights))
    # print(class_weights)
    # print(f'Weights computed in {t2 - t1} seconds')

    # weights = np.zeros((batch_size, patch_size, patch_size, n_classes), dtype=np.float32)
    # for cls_num in classes_weights:
    #     weights[...,cls_num] = np.full((patch_size, patch_size), classes_weights[cls_num], dtype=np.float32)
    # print(f'Weights: {classes_weights}')

    y_train = to_categorical(y_train, num_classes=n_classes).astype(np.uint8)
    y_test = to_categorical(y_test, num_classes=n_classes).astype(np.uint8)

    input_shape = (patch_size, patch_size, 3)
    initial_epoch = 0

    # custom_loss = functools.partial(losses.weighted_dice_loss, weights=weights)
    custom_loss = functools.partial(losses.cce_dice_loss)
    custom_loss.__name__ = 'cce_dice_loss'

    model = custom_unet(
        input_shape,
        n_classes=n_classes,
        filters=n_filters,
        use_batch_norm=True,
        n_layers=n_layers,
        output_activation='softmax'
    )

    model.compile(
        optimizer=Adam(), 
        loss = custom_loss,
        # loss = 'categorical_crossentropy',
        metrics=[iou]
    )

    if model_name:
        model = load_model(os.path.join(train_params["model_path"], model_name), custom_objects={'iou' : iou, 'weighted_dice_loss' : custom_loss})
        initial_epoch = int(model_name.split('_')[1])
        print(f'Model {model_name} loaded. Continue training from epoch {initial_epoch + 1}')

    # if model_name:
    #     model.load_weights(os.path.join(train_params["model_path"], model_name))
    #     model._make_train_function()
    #     with open(os.path.join(train_params["model_path"], model_name.replace('model', 'optimizer').replace('hdf5', 'pkl')), 'rb') as f:
    #         weight_values = pickle.load(f)
    #     model.optimizer.set_weights(weight_values)
    #     initial_epoch = int(model_name.split('_')[1])
    #     print(f'Model {model_name} loaded. Continue training from epoch {initial_epoch + 1}')


    callback_checkpoint = AdvancedCheckpoint(
        model=model,
        save_best_only=True,
        verbose=True,
        output_path=train_params["model_path"]
    )

    # Setting up callbacks
    ######################################
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
        patience=6,
        restore_best_weights=True
    )

    # callback_checkpoint = ModelCheckpoint(
    #     filepath=os.path.join(train_params["model_path"], 'model_{epoch:02d}_{loss:.2f}.hdf5'),
    #     verbose=1, 
    #     monitor='loss', 
    #     save_best_only=True,
    #     save_weights_only=False
    # )

    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.2,
        patience=4,
        min_lr=0.00001,
        verbose=1
    )

    csv_logger = CSVLogger('training.log')
    ######################################

    train_generator = PatchGenerator(images=x_train, masks=y_train, names=train_names, patch_size=patch_size, batch_size=batch_size, full_augment=train_params["full_augment"], balanced=True)

    steps_per_epoch = 10
    history = model.fit(
        iter(train_generator),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[callback_checkpoint, callback_test, csv_logger, reduce_lr, early_stop],
        initial_epoch=initial_epoch
    )

    plot_segm_history(history, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', action='store')
    args = parser.parse_args()

    if args.model == None:
        if os.path.exists(train_params["output_path"]):
            shutil.rmtree(train_params["output_path"])
    os.makedirs(train_params["model_path"], exist_ok=True)

    print('Training params:')
    print(train_params)

    train(n_classes=len(classes_mask.keys()), n_layers=train_params["n_layers"], 
            n_filters=train_params["n_filters"], epochs=train_params["epochs"], 
            path=train_params["dataset_path"], batch_size=train_params["batch_size"],
            patch_size=train_params["patch_size"], overlay=train_params["overlay"],
            output_path=train_params["output_path"], model_name=args.model)