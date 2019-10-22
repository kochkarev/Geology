import numpy as np
import matplotlib.pyplot as plt
from data_utils.load_data import get_pairs_from_paths

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

        seg_img = np.zeros_like( seg )

        for c in range(n_classes):
            seg_img[:,:,0] += ((seg[:,:,0] == c) * (colors[c][0])).astype('uint8')
            seg_img[:,:,1] += ((seg[:,:,0] == c) * (colors[c][1])).astype('uint8')
            seg_img[:,:,2] += ((seg[:,:,0] == c) * (colors[c][2])).astype('uint8')

        cv2.imshow("img" , cv2.resize(img, (0, 0), fx=0.2, fy=0.2))
        cv2.imshow("seg_img" , cv2.resize(seg_img, (0, 0), fx=0.2, fy=0.2))
        if cv2.waitKey(0) == 27: return
    cv2.destroyAllWindows()

def compare_masks_red(truth, pred):

    difference = cv2.subtract(truth, pred)

    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]

    return difference

def compare_masks_rgb(truth, pred):

    height = truth.shape[0]
    width = truth.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8) # BGR
    blank_image[:,:,1] = np.copy(truth[:,:,0]) # Green
    blank_image[:,:,2] = np.copy(pred[:,:,0]) # Red

    return blank_image
