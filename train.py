from data_utils.load_data import get_imgs_masks
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from models.unet import custom_unet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD

def train(num_classes):

    x, y = get_imgs_masks("")
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
        num_layers=2,
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
        loss = 'categorical_crossentropy'#,
        #metrics=[iou]
    )

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=1,
        epochs=20,
        validation_data=(x_val, y_val),
        callbacks=[callback_checkpoint]
    )