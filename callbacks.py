import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from utils import visualize_segmentation_result, plot_metrics_history, colorize_mask, plot_per_class_history, contrast_mask, plot_lrs
import numpy as np
from data_utils import combine_patches, split_to_patches
from metrics import calc_metrics
from PIL import Image
from statistics import mean
import os
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from time import time

class TestResults(Callback):

    def __init__(self, images, masks, model, n_classes, batch_size, patch_size, offset, output_path, all_metrics : list):

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
        self.all_metrics = all_metrics
        self.metrics_results = {i : dict() for i in all_metrics}
        self.metrics_per_cls_res = {i : [] for i in range(n_classes)}
        self.lrs = []

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
            pred_patches.extend(np.squeeze(np.split(prediction, self.batch_size)))
        
        pred_patches = pred_patches[:init_patch_len]
        result = combine_patches(pred_patches, self.patch_size, self.offset, (new_size[0], new_size[1]), (height, width))
        return result        

    
    def on_epoch_end(self, epoch, logs=None):

        self.lrs.append(K.eval(self.model.optimizer.lr))

        predicted = []
        all_metrics = ['iou']
        metrics_values = {i : 0 for i in all_metrics}
        tmp_metrics_per_cls_res = {i : [] for i in range(self.n_classes)}
        t1 = time()
        ii = 0
        for image, mask in zip(self.images, self.masks):
            print(f'Testing on {ii+1} / {self.images.shape[0]}')
            pred = self.predict_image(image)
            print(pred.shape)
            predicted.append(pred)
            metrics = calc_metrics(mask[self.offset:-self.offset,self.offset:-self.offset,...], 
                                        pred[self.offset:-self.offset,self.offset:-self.offset,...], all_metrics, self.n_classes)
            for metric in metrics:
                print('Metrics for each class:')
                i = 0
                for value in metric[1]:
                    print("{} : {}".format(metric[0], value))
                    tmp_metrics_per_cls_res[i].append(value)
                    i +=1
                metrics_values[metric[0]] += sum(metric[1]) / len(metric[1])
            ii += 1

        for i in range(self.n_classes):
            self.metrics_per_cls_res[i].append(mean(tmp_metrics_per_cls_res[i]))

        print('Average metrics values:')
        for metrics_name in metrics_values.keys():
            self.metrics_results[metrics_name][epoch+1] = metrics_values[metrics_name] / self.images.shape[0]
            print('{name} : {val}'.format(name=metrics_name, val=(metrics_values[metrics_name] / self.images.shape[0])))

        t2 = time()
        print(f'prediction completed in {t2-t1} seconds')

        print('Processing visualization:')
        visualize_segmentation_result(self.images, [np.argmax(i, axis=2) for i in self.masks], [np.argmax(i, axis=2) for i in predicted],
                                      n_classes=self.n_classes, output_path=self.output_path, epoch=epoch)
        t3 = time()
        print(f'Visualization completed in {t3-t2} seconds')

    def on_train_end(self, logs=None):
        plot_metrics_history(self.metrics_results)
        plot_per_class_history(self.metrics_per_cls_res)
        plot_lrs(self.lrs)

# class LearningRateDecay:
# 	def plot(self, epochs, title="Learning Rate Schedule"):
# 		# compute the set of learning rates for each corresponding
# 		# epoch
# 		lrs = [self(i) for i in epochs]
# 		# the learning rate schedule
# 		plt.style.use("ggplot")
# 		plt.figure()
# 		plt.plot(epochs, lrs)
# 		plt.title(title)
# 		plt.xlabel("Epoch #")
# 		plt.ylabel("Learning Rate")

# class StepDecay(LearningRateDecay):
# 	def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
# 		# store the base initial learning rate, drop factor, and
# 		# epochs to drop every
# 		self.initAlpha = initAlpha
# 		self.factor = factor
# 		self.dropEvery = dropEvery
# 	def __call__(self, epoch):
# 		# compute the learning rate for the current epoch
# 		exp = np.floor((1 + epoch) / self.dropEvery)
# 		alpha = self.initAlpha * (self.factor ** exp)
# 		# return the learning rate
# 		return float(alpha)