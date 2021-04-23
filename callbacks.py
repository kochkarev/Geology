from pathlib import Path
from time import time

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

from metrics import calc_metrics
from utils.vis import plot_lrs, plot_metrics, vis_segmentation


class TestResults(Callback):
    def __init__(self, images, masks, names, predict_func, n_classes, batch_size, patch_size, patch_overlay, offset,
                 output_path: Path, all_metrics: list, vis: bool):
        self.images = images
        self.masks = masks
        self.names = names
        self.predict_func = predict_func
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.overlay = patch_overlay
        self.offset = offset
        self.output_path = output_path
        self.all_metrics = all_metrics
        self.vis = vis
        self.metrics_acc = {i: [] for i in all_metrics}
        self.lrs = []

    def _average_metrics(self, metrics):
        n_imgs = len(metrics)
        lh = len(metrics[0])
        return [sum(m[f] for m in metrics) / n_imgs for f in range(lh)]

    def _print_per_class_metric(self, metric_name, metric_val, log):
        n = len(metric_val)
        for i in range(n - 1):
            s = f'{metric_name} for class {i}: {metric_val[i]:.5f}'
            print(s)
            log.write(s + '\n')
        s = f'{metric_name} for all classes: {metric_val[-1]:.5f}'
        print(s), log.write(s + '\n')

    def on_epoch_end(self, epoch, logs=None):
        output_path_name = self.output_path / f'epoch_{epoch+1}'
        output_path_name.mkdir(exist_ok=True)
        metrics_log_name = output_path_name / 'metrics.txt'
        log = open(metrics_log_name, "a+")
        
        self.lrs.append(K.eval(self.model.optimizer.lr))

        # --- do predictions ---

        preds = []
        metrics_per_image = {m: [] for m in self.all_metrics}
        t1 = time()
        ii = 0
        for image, mask in zip(self.images, self.masks):
            s = f'Testing on {ii+1} of {len(self.images)}'
            print('\n' + s), log.write('\n' + s + '\n')
            # --- do prediction ---
            pred = self.predict_func(image)
            preds.append(pred)
            # --- calc metrics ---
            metric_maps = calc_metrics(mask, pred, self.all_metrics, self.offset)
            s = 'Metrics:'
            print('\n' + s), log.write(s + '\n')
            for metric_name, metric_vals in metric_maps.items():
                self._print_per_class_metric(metric_name, metric_vals, log)
                metrics_per_image[metric_name].append(metric_vals)
            ii += 1

        s = 'Average metrics values:'
        print(s), log.write('\n' + s + '\n')
        for metric_name, metric_vals in metrics_per_image.items():
            metric_avg = self._average_metrics(metric_vals)
            self.metrics_acc[metric_name].append(metric_avg)
            self._print_per_class_metric(metric_name, metric_avg, log)

        print('{"metric": "iou_all_test", "value":' + str(metric_avg[-1]) + ', "epoch": ' + str(epoch + 1) + '}')

        t2 = time()
        print(f'Prediction completed in {t2-t1} seconds')

        print('Processing visualization:')
        if self.vis:
            for i, image in enumerate(self.images):
                img = (image * 255).astype(np.uint8)
                mask = np.argmax(self.masks[i], axis=-1)
                pred = np.argmax(preds[i], axis=-1).astype(np.uint8)
                vis_segmentation(img, mask, pred, self.n_classes, self.offset, output_path=self.output_path, epoch=epoch, img_num=i)
        
        for metric_name in self.all_metrics:
            plot_metrics(self.metrics_acc[metric_name], metric_name, self.output_path)
        plot_lrs(self.lrs, self.output_path)
        t3 = time()
        print(f'Visualization completed in {t3-t2} seconds')
