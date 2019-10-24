from data_utils.load_data import get_imgs_masks, resize_imgs_masks
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from models.unet import custom_unet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from metrics import iou
from utils import plot_segm_history
import os
import numpy as np
from callbacks import VisualizeResults

def train(num_classes, num_layers, path, epochs, show_history=True):

    x, y = get_imgs_masks(path)
    print("Found {num} images and {num1} masks".format(num=len(x), num1=len(y)))
    x, y = resize_imgs_masks(num_layers, x, y)
    print("After resize {num} images and {num1} masks".format(num=len(x), num1=len(y)))
    x = np.asarray(x, dtype=np.float32) / 255 
    y = np.asarray(y, dtype=np.uint8)
<<<<<<< HEAD

    print(x.shape)
    print(y.shape)
=======
>>>>>>> e2f5b6f412fad936ae3bfebbbb787051cb7a4425

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=0)
    print("Training: {x_tr} images and {y_tr} masks".format(x_tr=x_train.shape, y_tr=y_train.shape))
    print("Validation: {x_v} images and {y_v} masks".format(x_v=x_val.shape, y_v=y_val.shape))
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    print("After transforming masks: train: {tr}; validation: {val}".format(tr=y_train.shape, val=y_val.shape))
    train_gen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )

    input_shape = x_train[0].shape

    model = custom_unet(
        input_shape,
        num_classes=num_classes,
        filters=16,
        use_batch_norm=True,
        dropout=0.0,
        dropout_change_per_layer=0.0,
        num_layers=num_layers,
        output_activation='softmax'
    )

    # model_filename = 'segm_model_v1.h5'
    # callback_checkpoint = ModelCheckpoint(
    #     model_filename, 
    #     verbose=1, 
    #     monitor='val_loss', 
    #     save_best_only=True,
    # )

    model.compile(
        optimizer=Adam(), 
        loss = 'categorical_crossentropy',
        metrics=[iou]
    )

    # callback_visualize = VisualizeResults(
    #     images=x_val, 
    #     masks=y_val, 
    #     model=model, 
    #     n_classes=num_classes
    # )

    history = model.fit_generator(
        train_gen.flow(x_train, y_train, batch_size=2),
        steps_per_epoch=len(x_train) / 2,
        epochs=epochs,
<<<<<<< HEAD
        validation_data=(x_val, y_val)
        #callbacks=[callback_checkpoint, callback_visualize]
=======
        validation_data=(x_val, y_val),
        #callbacks=[callback_checkpoint, callback_visualize]
        callbacks=[callback_checkpoint]
>>>>>>> e2f5b6f412fad936ae3bfebbbb787051cb7a4425
    )

    if show_history:
        plot_segm_history(history)

if __name__ == "__main__":
    path = os.path.join("input", "dataset", "*_NEW.png")
    train(num_classes=4, num_layers=2, epochs=20, path=path)