import tensorflow as tf
from data_utils import get_imgs_masks, resize_imgs_masks
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from unet import custom_unet
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam, SGD
from metrics import iou, iou_multiclass
from utils import plot_segm_history
import os
import numpy as np
from callbacks import TestResults
from generators import PatchGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from data_utils import combine_patches, split_to_patches
from PIL import Image
from utils import visualize_prediction_result
from data_utils import get_unmarked_images

class Prediction:

    def __init__(self, model_path, output_path, patch_size, batch_size, offset):
        self.model = load_model(model_path, custom_objects={"iou":iou})
        self.model.compile(
            optimizer=Adam(), 
            loss = 'categorical_crossentropy',
            metrics=[iou]
        )
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.offset = offset
        self.output_path = output_path

    def __predict_image__(self, img):
        height = img.shape[0]
        width = img.shape[1]

        patches, new_size = split_to_patches(img, self.patch_size, self.offset)
        init_patch_len = len(patches)

        while (len(patches) % self.batch_size != 0):
            patches.append(patches[-1])
        pred_patches = []

        for i in range(0, len(patches), self.batch_size):

            batch = np.stack(patches[i : i+self.batch_size])
            prediction = self.model.predict_on_batch(batch)

            for x in prediction:
                pred_patches.append(x)
        
        pred_patches = pred_patches[:init_patch_len]
        result = combine_patches(pred_patches, self.patch_size, self.offset, (new_size[0], new_size[1]), (height, width))

        return result

    def predict(self, images : list):

        for image_name in images:

            image = np.array(Image.open(image_name))
            predicted = self.__predict_image__(image)
            visualize_prediction_result(image, np.argmax(predicted, axis=2), os.path.basename(image_name), output_path=self.output_path)

if __name__ == "__main__":

    output_path = os.path.join("output", "prediction")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pred = Prediction("model_v1_3lay.h5", output_path, 512, 8, 2*3)

    unmarked = get_unmarked_images(os.path.join("input", "UMNIK_2019", "BoxA_DS1", "img"), os.path.join("input", "dataset"))
    pred.predict(unmarked)