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
    x, y = resize_imgs_masks(num_layers, x, y)

    x = np.asarray(x, dtype=np.float32) / 255 
    y = np.asarray(y, dtype=np.float32)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=0)
    
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    train_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    input_shape = x_train[0].shape

    model = custom_unet(
        input_shape,
        num_classes=num_classes,
        filters=32,
        use_batch_norm=True,
        dropout=0.0,
        dropout_change_per_layer=0.0,
        num_layers=num_layers,
        output_activation='softmax'
    )

    model_filename = 'segm_model_v1.h5'
    callback_checkpoint = ModelCheckpoint(
        model_filename, 
        verbose=1, 
        monitor='val_loss', 
        save_best_only=True,
    )

    model.compile(
        optimizer=Adam(), 
        loss = 'categorical_crossentropy',
        metrics=[iou]
    )

    callback_visualize = VisualizeResults(
        images=x_val, 
        masks=y_val, 
        model=model, 
        n_classes=num_classes
    )

    history = model.fit_generator(
        train_gen.flow(x_train, y_train, batch_size=4),
        steps_per_epoch=len(x_train) / 4,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[callback_checkpoint, callback_visualize]
    )

    if show_history:
        plot_segm_history(history)

if __name__ == "__main__":
    path = os.path.join("input", "dataset")
    train(num_classes=4, num_layers=2, epochs=20, path=path)