#import plaidml.keras
#plaidml.keras.install_backend()
import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_pairs_from_paths
import os
from PIL import Image

def plot_segm_history(history, metrics=['iou', 'val_iou'], losses=['loss', 'val_loss']):
    # summarize history for iou
    plt.figure(figsize=(12,6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle('metrics over epochs', fontsize=20)
    plt.ylabel('metric', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    #plt.yticks(np.arange(0.3, 1, step=0.02), fontsize=35)
    #plt.xticks(fontsize=35)
    plt.legend(metrics, loc='center right', fontsize=15)
    plt.show()
    # summarize history for loss
    plt.figure(figsize=(12,6))    
    for loss in losses:
        plt.plot(history.history[loss], linewidth=3)
    plt.suptitle('loss over epochs', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    #plt.yticks(np.arange(0, 0.2, step=0.005), fontsize=35)
    #plt.xticks(fontsize=35)
    plt.legend(losses, loc='center right', fontsize=15)
    plt.show()

import random
import cv2

random.seed(0)
class_colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(5000)]

def visualize_segmentation_dataset(path, n_classes):

    img_seg_pairs = get_pairs_from_paths(path)

    colors = class_colors

    print("Press any key to navigate. ")
    for im_fn , seg_fn in img_seg_pairs :

        img = cv2.imread(im_fn)
        seg = cv2.imread(seg_fn)
        print("Found the following classes" , np.unique(seg))

        seg_img = np.zeros_like(seg)

        for c in range(n_classes):
            seg_img[:,:,0] += ((seg[:,:,0] == c) * (colors[c][0])).astype('uint8')
            seg_img[:,:,1] += ((seg[:,:,0] == c) * (colors[c][1])).astype('uint8')
            seg_img[:,:,2] += ((seg[:,:,0] == c) * (colors[c][2])).astype('uint8')

        cv2.imshow("img" , cv2.resize(img, (0, 0), fx=0.2, fy=0.2))
        cv2.imshow("seg_img" , cv2.resize(seg_img, (0, 0), fx=0.2, fy=0.2))
        if cv2.waitKey(0) == 27: return
    cv2.destroyAllWindows()

def colorize_mask(mask, n_classes):

    color_mask = np.zeros_like(mask)
    colors = class_colors

    for c in range(n_classes):
        if mask.ndim == 3:
            color_mask[:,:,0] += ((mask[:,:,0] == c) * (colors[c][0])).astype('uint8')
            color_mask[:,:,1] += ((mask[:,:,0] == c) * (colors[c][1])).astype('uint8')
            color_mask[:,:,2] += ((mask[:,:,0] == c) * (colors[c][2])).astype('uint8')
        elif mask.ndim == 2:
            color_mask += ((mask == c) * (colors[c][2])).astype('uint8')

    return color_mask

def compare_masks_red(truth, pred):

    difference = cv2.subtract(truth, pred)

    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]

    return difference

def compare_masks_rgb(truth, pred):

    max1 = np.amax(truth)
    max2 = np.amax(pred)

    height = truth.shape[0]
    width = truth.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8) # BGR
    blank_image[:,:,1] = np.copy(np.multiply(truth[:,:,0], 255 / max1)) # Green
    blank_image[:,:,2] = np.copy(np.multiply(pred[:,:,0], 255 / max2)) # Red

    return blank_image

def create_error_mask(img : np.ndarray, pred : np.ndarray, num_classes : int = 4):

    assert (img.ndim == 2 and pred.ndim == 2), ('Expected (H x W) masks, got img with shape {} and pred with shape {}'.format(img.shape, pred.shape))

    masks = []
    for i in range(num_classes):
        masks.append(np.array(np.where(img == i, 1, 0) * np.where(pred == i, 1, 0), dtype=np.uint8))
    res = np.zeros_like(img, dtype=np.uint8)
    for mask in masks:
        res += mask

    return res

def visualize_error_mask(mask : np.ndarray, show=False):

    assert (mask.ndim == 2), ('Expected (H x W) output')

    mask = np.asarray(np.dstack((mask, mask, mask)), dtype=np.uint8)
    mask = np.array(np.where(mask == (0,0,0), (255,0,0), (0,255,0)), dtype=np.uint8)

    if show:
        Image.fromarray(mask).show()

    return mask

def visualize_segmentation_result(images, masks, preds=None, figsize=4, nm_img_to_plot=2, n_classes=4, ouput_path=None, epoch=0):

    cols = 2 if preds is None else 5

    fig, axes = plt.subplots(nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize))
    axes[0, 0].set_title("original", fontsize=15) 
    axes[0, 1].set_title("ground truth", fontsize=15)
    if not (preds is None):
        axes[0, 2].set_title("prediction", fontsize=15) 
        axes[0, 3].set_title("error map", fontsize=15) 
        axes[0, 4].set_title("overlay", fontsize=15)
    
    im_id = 0
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(images[im_id])
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(colorize_mask(masks[im_id], n_classes=n_classes))
        axes[m, 1].set_axis_off()        
        if not (preds is None):
            axes[m, 2].imshow(colorize_mask(preds[im_id], n_classes=n_classes))
            axes[m, 2].set_axis_off()
            axes[m, 3].imshow(visualize_error_mask(create_error_mask(masks[im_id], preds[im_id], num_classes=n_classes)))
            axes[m, 3].set_axis_off()
            axes[m, 4].imshow(images[im_id])
            axes[m, 4].imshow(visualize_error_mask(create_error_mask(masks[im_id], preds[im_id], num_classes=n_classes)), alpha=0.5)
            axes[m, 4].set_axis_off()
        im_id += 1

    #plt.show()

    if (ouput_path != None):
        output_name = os.path.join(ouput_path, str(epoch) + '_EPOCH.jpg')
        fig.savefig(output_name)
    

