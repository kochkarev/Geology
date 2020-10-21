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
import losses
import functools
from config import classes_weights, train_params, classes_mask
from time import time
from unet import custom_unet

class Prediction:

    def __init__(self, model_path, output_path, patch_size, batch_size, offset, n_classes, n_filters, n_layers):
        weights = np.zeros((batch_size, patch_size, patch_size, 4), dtype=np.float32)
        for cls_num in classes_weights:
            weights[...,cls_num] = np.full((patch_size, patch_size), classes_weights[cls_num], dtype=np.float32)
        custom_loss = functools.partial(losses.weighted_dice_loss, weights=weights)
        custom_loss.__name__ = 'weighted_dice_loss'
        
        self.model = custom_unet(
            (patch_size, patch_size, 3),
            n_classes=n_classes,
            filters=n_filters,
            use_batch_norm=True,
            n_layers=n_layers,
            output_activation='softmax'
        )
        self.model.compile(
            optimizer=Adam(), 
            loss = custom_loss,
            metrics=[iou]
        )
        #  custom_objects={"iou":iou, "weighted_dice_loss":custom_loss}
        self.model.load_weights(model_path)

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.offset = offset
        self.output_path = output_path

    def __predict_image__(self, img):

        patches = split_to_patches(img, self.patch_size, self.offset, overlay=0.25)
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
        result = combine_patches(pred_patches, self.patch_size, self.offset, overlay=0.25, orig_shape=(img.shape[0], img.shape[1], pred_patches[0].shape[2]))

        return result

    def predict(self, images : list):

        for image_name in images:

            image = np.array(Image.open(image_name))
            predicted = self.__predict_image__(image)
            
            # t1=time()
            # L = np.argsort(predicted, axis=-1)
            # max1_idx, max2_idx = L[...,-1], L[...,-2]
            # I, J = np.ogrid[:predicted.shape[0],:predicted.shape[1]]
            # res = np.argmax(predicted, axis=-1)
            # max_values1 = predicted[I,J,max1_idx]
            # max_values2 = predicted[I,J,max2_idx]
            # res[np.abs(max_values1 - max_values2) <= 0.2] = 255
            # t2=time()
            # print(f'{t2-t1} seconds')
            # t1=time()
            # res1 = np.argmax(predicted, axis=-1)
            # t2=time()
            # print(f'argmax {t2-t1} seconds')

            visualize_prediction_result(image, np.argmax(predicted, axis=2), os.path.basename(image_name), output_path=self.output_path)
            # visualize_prediction_result(image, res, os.path.basename(image_name), output_path=self.output_path)

if __name__ == "__main__":

    output_path = os.path.join("output", "prediction")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pred = Prediction(model_path="/Users/a17641623/Documents/Geology/models/model_46_0.07.hdf5", 
                        output_path=output_path,
                        patch_size=train_params["patch_size"],
                        batch_size=train_params["batch_size"],
                        offset=2*train_params["n_layers"],
                        n_classes=len(classes_mask.keys()),
                        n_filters=train_params["n_filters"],
                        n_layers=train_params["n_layers"])

    # unmarked = get_unmarked_images(os.path.join("input", "UMNIK_2019", "BoxA_DS1", "img"), os.path.join("input", "dataset"))
    # unmarked = [f"/Users/a17641623/Documents/Geology/test_img1/{i+1}.jpg" for i in range(4)]
    # unmarked += ["/Users/a17641623/Documents/Geology/test_img/5.png"]
    unmarked = [
        "/Users/a17641623/Documents/Geology/input/dataset/Py-Cpy-Sh-GL26.jpg",
        "/Users/a17641623/Documents/Geology/input/dataset/Py-Sh-GL49.jpg",
        "/Users/a17641623/Documents/Geology/input/dataset/Py-Sh-BR-Gl47.jpg",
        "/Users/a17641623/Documents/Geology/input/dataset/Py-Sh-GL21.jpg",
        "/Users/a17641623/Documents/Geology/input/dataset/Py-Cpy-Sh-GL3.jpg",
        "/Users/a17641623/Documents/Geology/input/dataset/Py-Cpy-Sh-BR-29.jpg",
        "/Users/a17641623/Documents/Geology/input/dataset/Py-Cpy-Sh-BR-GL30.jpg",
        "/Users/a17641623/Documents/Geology/input/dataset/Py-Cpy-Sh-GL31.jpg",
        "/Users/a17641623/Documents/Geology/input/dataset/Py-Cpy-Sh-BR-28.jpg",
        "/Users/a17641623/Documents/Geology/input/dataset/Py-Cpy-Sh-BR-GL27.jpg"
    ]
    print(unmarked)
    pred.predict(unmarked)