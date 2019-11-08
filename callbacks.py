#import plaidml.keras
#plaidml.keras.install_backend()
from keras.callbacks import Callback
from utils import visualize_segmentation_result
import numpy as np
from data_utils import combine_patches, split_to_patches
from metrics import calc_metrics

class TestResults(Callback):

    def __init__(self, images, masks, model, n_classes, batch_size, patch_size, offset, output_path, no_split : bool):

        self.images = images
        self.masks = masks
        self.model = model
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.offset = offset
        self.output_path = output_path
        self.no_split = no_split

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
            prediction = self.model.predict_on_batch(batch)

            for x in prediction:
                pred_patches.append(x)
        
        pred_patches = pred_patches[:init_patch_len]
        result = combine_patches(pred_patches, self.patch_size, self.offset, (new_size[0], new_size[1]), (height, width))

        return result        

    
    def on_epoch_end(self, epoch, logs=None):

        predicted = []
        all_metrics = ['iou']
        if self.no_split:
            all_metrics_no_split = [i for i in all_metrics]
        metrics_values = {i : 0 for i in all_metrics}
        if self.no_split:
            metrics_values_no_split = {i : 0 for i in all_metrics}
        print('Calculating metrics:')
        for image, mask in zip(self.images, self.masks):

            pred = self.predict_image(image)
            predicted.append(pred)
            assert (pred.shape == mask.shape), ('Something bad')
            metrics = calc_metrics(np.argmax(mask[self.offset:-self.offset,self.offset:-self.offset,...], axis=2), 
                                        np.argmax(pred[self.offset:-self.offset,self.offset:-self.offset,...], axis=2), all_metrics)
            if self.no_split:
                pred_no_split = self.model.predict(image)
                metrics_no_split = calc_metrics(np.argmax(mask[self.offset:-self.offset,self.offset:-self.offset,...], axis=2), 
                                        np.argmax(pred_no_split[self.offset:-self.offset,self.offset:-self.offset,...], axis=2), all_metrics_no_split)
            for metric in metrics:
                metrics_values[metric[0]] += metric[1]
            if self.no_split:
                for metric in metrics_no_split:
                    metrics_values_no_split[metric[0]] += metric[1]

        for metrics_name in metrics_values.keys():
            print('{name} : {val}'.format(name=metrics_name, val=(metrics_values[metrics_name] / self.images.shape[0])))
        if self.no_split:
            print('Metrics values for image w/o splitting to patches:')
            for metrics_name in metrics_values_no_split.keys():
                print('{name} : {val}'.format(name=metrics_name, val=(metrics_values_no_split[metrics_name] / self.images.shape[0])))

        
        print('Processing visualization:')
        visualize_segmentation_result(self.images, [np.argmax(i, axis=2) for i in self.masks], [np.argmax(i, axis=2) for i in predicted], 
                                    figsize=6, nm_img_to_plot=len(predicted), n_classes=self.n_classes, ouput_path=self.output_path, epoch=epoch)
        print('Visualization results saved in {} directory'.format(self.output_path))