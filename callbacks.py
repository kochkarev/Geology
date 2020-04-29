import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from utils import visualize_segmentation_result, plot_metrics_history, colorize_mask, plot_per_class_history, contrast_mask, plot_lrs, visualize_pred_heatmaps
import numpy as np
from data_utils import combine_patches, split_to_patches
from metrics import calc_metrics
from PIL import Image
from statistics import mean
import os
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from time import time
from config import classes_mask, train_params
from datetime import datetime
import pickle

class TestResults(Callback):
    def __init__(self, images, masks, names, model, n_classes, batch_size, patch_size, overlay, offset, output_path, all_metrics : list):
        self.images = images
        self.masks = masks
        self.names = names
        self.model = model
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.overlay = overlay
        self.offset = offset
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.all_metrics = all_metrics
        self.metrics_results = {i : dict() for i in all_metrics}
        self.metrics_per_cls_res = {i : [] for i in range(n_classes)}
        self.lrs = []

    def predict_image(self, img):
        patches = split_to_patches(img, self.patch_size, self.offset, overlay = 0.25)
        init_patch_len = len(patches)

        while (len(patches) % self.batch_size != 0):
            patches.append(patches[-1])
        pred_patches = []

        for i in range(0, len(patches), self.batch_size):
            batch = np.stack(patches[i : i+self.batch_size])
            prediction = self.model.predict_on_batch(batch)
            pred_patches.extend(np.squeeze(np.split(prediction, self.batch_size)))
        
        pred_patches = pred_patches[:init_patch_len]
        result = combine_patches(pred_patches, self.patch_size, self.offset, overlay=0.25, orig_shape=(img.shape[0], img.shape[1], pred_patches[0].shape[2]))
        print(f'in predict: {np.min(result)} : {np.max(result)}')
        return result
    
    def on_epoch_end(self, epoch, logs=None):
        output_path_name = os.path.join(self.output_path, f'epoch_{epoch+1}')
        os.makedirs(output_path_name, exist_ok=True)
        metrics_log_name = os.path.join(output_path_name, "metrics.txt")
        if os.path.exists(metrics_log_name):
            os.remove(metrics_log_name)
        metrics_log = open(metrics_log_name, "a+")
        
        self.lrs.append(K.eval(self.model.optimizer.lr))

        predicted = []
        all_metrics = ['iou']
        metrics_values = {i : 0 for i in all_metrics}
        tmp_metrics_per_cls_res = {i : [] for i in range(self.n_classes)}
        t1 = time()
        ii = 0
        for image, mask in zip(self.images, self.masks):
            s = f'Testing on {ii+1} of {self.images.shape[0]}'
            print(s)
            metrics_log.write('\n' + s + '\n')
            pred = self.predict_image(image)
            predicted.append(pred)
            metrics = calc_metrics(mask[self.offset:-self.offset,self.offset:-self.offset,...], 
                                        pred, all_metrics, self.n_classes)
            for metric in metrics:
                s = 'Metrics for each class:'
                print(s)
                metrics_log.write(s + '\n')
                i = 0
                for value in metric[1]:
                    s = f"{metric[0]} for {classes_mask[i]} : {value}"
                    print(s)
                    metrics_log.write(s + '\n')
                    tmp_metrics_per_cls_res[i].append(value)
                    i +=1
                metrics_values[metric[0]] += sum(metric[1]) / len(metric[1])
            ii += 1

        for i in range(self.n_classes):
            self.metrics_per_cls_res[i].append(mean(tmp_metrics_per_cls_res[i]))

        s = 'Average metrics values:'
        print(s)
        metrics_log.write('\n' + s + '\n')
        for metrics_name in metrics_values.keys():
            self.metrics_results[metrics_name][epoch+1] = metrics_values[metrics_name] / self.images.shape[0]
            s = f'{metrics_name} : {(metrics_values[metrics_name] / self.images.shape[0])}'
            print(s)
            metrics_log.write(s + '\n')

        t2 = time()
        print(f'Prediction completed in {t2-t1} seconds')

        print('Processing visualization:')
        visualize_segmentation_result(np.array([i[self.offset:-self.offset,self.offset:-self.offset,...] for i in self.images]),
                    [np.argmax(i[self.offset:-self.offset,self.offset:-self.offset,...], axis=2) for i in self.masks],
                    [np.argmax(i, axis=2) for i in predicted], names=self.names, n_classes=self.n_classes, output_path=self.output_path, epoch=epoch)
        visualize_pred_heatmaps(predicted, self.n_classes, self.output_path, epoch)
        plot_metrics_history(self.metrics_results, self.output_path)
        plot_per_class_history(self.metrics_per_cls_res, self.output_path)
        plot_lrs(self.lrs, self.output_path)
        t3 = time()
        print(f'Visualization completed in {t3-t2} seconds')

    def on_train_begin(self, logs=None):
        params_name = os.path.join(self.output_path, "train_params.txt")
        if os.path.exists(params_name):
            os.remove(params_name)
        train_log = open(params_name, "a+")

        train_log.write(f'Training begin at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n\n')

        for param in train_params:
            train_log.write(param + '=' + str(train_params[param]) + '\n')