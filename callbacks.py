#import plaidml.keras
#plaidml.keras.install_backend()
from keras.callbacks import Callback
from utils import visualize_segmentation_result, plot_metrics_history, colorize_mask, plot_per_class_history, contrast_mask
import numpy as np
from data_utils import combine_patches, split_to_patches
from metrics import calc_metrics
from PIL import Image
from statistics import mean
import os

class TestResults(Callback):

    def __init__(self, images, masks, model, n_classes, batch_size, patch_size, offset, output_path, no_split : bool, all_metrics : list):

        self.images = images
        self.masks = masks
        self.model = model
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.offset = offset
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.no_split = no_split
        self.all_metrics = all_metrics
        self.metrics_results = {i : dict() for i in all_metrics}
        self.metrics_per_cls_res = {i : [] for i in range(n_classes)}

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
        # if self.no_split:
        #     all_metrics_no_split = [i for i in all_metrics]
        metrics_values = {i : 0 for i in all_metrics}
        tmp_metrics_per_cls_res = {i : [] for i in range(self.n_classes)}
        # if self.no_split:
        #     metrics_values_no_split = {i : 0 for i in all_metrics}
        print('Testing current model:')
        ii = 0
        for image, mask in zip(self.images, self.masks):

            print('Predicting:')
            pred = self.predict_image(image)
            predicted.append(pred)
            print('Calculating metrics:')
            metrics = calc_metrics(mask[self.offset:-self.offset,self.offset:-self.offset,...], 
                                        pred[self.offset:-self.offset,self.offset:-self.offset,...], all_metrics, self.n_classes)
            # if self.no_split:
            #     pred_no_split = self.model.predict(image)
            #     metrics_no_split = calc_metrics(np.argmax(mask[self.offset:-self.offset,self.offset:-self.offset,...], axis=2), 
            #           np.argmax(pred_no_split[self.offset:-self.offset,self.offset:-self.offset,...], axis=2), all_metrics_no_split)
            for metric in metrics:
                print('Metrics for each class:')
                i = 0
                for value in metric[1]:
                    print("{} : {}".format(metric[0], value))
                    tmp_metrics_per_cls_res[i].append(value)
                    i +=1
                metrics_values[metric[0]] += sum(metric[1]) / len(metric[1])

            # if self.no_split:
            #     for metric in metrics_no_split:
            #         metrics_values_no_split[metric[0]] += metric[1]

            ii += 1

        for i in range(self.n_classes):
            self.metrics_per_cls_res[i].append(mean(tmp_metrics_per_cls_res[i]))

        print('Average metrics values:')
        for metrics_name in metrics_values.keys():
            self.metrics_results[metrics_name][epoch+1] = metrics_values[metrics_name] / self.images.shape[0]
            print('{name} : {val}'.format(name=metrics_name, val=(metrics_values[metrics_name] / self.images.shape[0])))
        # if self.no_split:
        #     print('Metrics values for image w/o splitting to patches:')
        #     for metrics_name in metrics_values_no_split.keys():
        #         print('{name} : {val}'.format(name=metrics_name, val=(metrics_values_no_split[metrics_name] / self.images.shape[0])))

        
        print('Processing visualization:')
        visualize_segmentation_result(self.images, [np.argmax(i, axis=2) for i in self.masks], [np.argmax(i, axis=2) for i in predicted], 
                                    figsize=6, nm_img_to_plot=len(predicted), n_classes=self.n_classes, output_path=self.output_path, epoch=epoch)
        print('Visualization results saved in {} directory'.format(self.output_path))

    def on_train_end(self, logs=None):

        plot_metrics_history(self.metrics_results)
        plot_per_class_history(self.metrics_per_cls_res)