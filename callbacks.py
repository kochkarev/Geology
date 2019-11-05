#import plaidml.keras
#plaidml.keras.install_backend()
from keras.callbacks import Callback
from utils import visualize_segmentation_result
import numpy as np
from data_utils import combine_patches, split_to_patches

class TestResults(Callback):

    def __init__(self, images, masks, model, n_classes, batch_size, patch_size, offset, output_path):

        self.images = images
        self.masks = masks
        self.model = model
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.offset = offset
        self.output_path = output_path

    def predict_image(self, img):

        height = img.shape[0]
        width = img.shape[1]

        patches, new_size = split_to_patches(img, self.patch_size, self.offset)
        init_patch_len = len(patches)

        while (len(patches) % self.batch_size != 0):
            patches.append(patches[-1])
        pred_patches = []

        for i in range(0, len(patches), self.batch_size):

            batch = np.stack(patches[i : i+self.batch_size])
            print(batch.shape)
            prediction = batch

            for x in prediction:
                pred_patches.append(x)
        
        pred_patches = pred_patches[:init_patch_len]
        result = combine_patches(pred_patches, self.patch_size, self.offset, (new_size[0], new_size[1]), (height, width))

        return result        

    
    def on_epoch_end(self, epoch, logs=None):

        for image in self.images:

            pred = self.predict_image(image)
            print('{} {}'.format(image.shape, pred.shape))

        #preds = self.model.predict(self.images)
        #visualize_segmentation_result(self.images, self.masks, preds, figsize=4, nm_img_to_plot=100, 
        #                                n_classes=self.n_classes, ouput_path=self.output_path, epoch=epoch)
        
