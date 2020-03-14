#import plaidml.keras
#plaidml.keras.install_backend()
import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_pairs_from_paths
import os
from PIL import Image
import shutil

def plot_segm_history(history, output_path, metrics=['iou', 'val_iou'], losses=['loss', 'val_loss']):
    # summarize history for iou
    fig1 = plt.figure(figsize=(12,6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle('metrics over epochs', fontsize=20)
    plt.ylabel('metric', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend(metrics, loc='center right', fontsize=15)
    fig1.savefig(os.path.join(output_path, 'metrics.jpg'))
    plt.show()
    # summarize history for loss
    fig2 = plt.figure(figsize=(12,6))    
    for loss in losses:
        plt.plot(history.history[loss], linewidth=3)
    plt.suptitle('loss over epochs', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend(losses, loc='center right', fontsize=15)
    fig2.savefig(os.path.join(output_path, 'loss.jpg'))
    plt.show()

import random
import cv2

random.seed(0)
class_colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(5000)]

# def visualize_segmentation_dataset(path, n_classes):

#     img_seg_pairs = get_pairs_from_paths(path)

#     colors = class_colors

#     print("Press any key to navigate. ")
#     for im_fn , seg_fn in img_seg_pairs :

#         img = cv2.imread(im_fn)
#         seg = cv2.imread(seg_fn)
#         print("Found the following classes" , np.unique(seg))

#         seg_img = np.zeros_like(seg)

#         for c in range(n_classes):
#             seg_img[:,:,0] += ((seg[:,:,0] == c) * (colors[c][0])).astype('uint8')
#             seg_img[:,:,1] += ((seg[:,:,0] == c) * (colors[c][1])).astype('uint8')
#             seg_img[:,:,2] += ((seg[:,:,0] == c) * (colors[c][2])).astype('uint8')

#         cv2.imshow("img" , cv2.resize(img, (0, 0), fx=0.2, fy=0.2))
#         cv2.imshow("seg_img" , cv2.resize(seg_img, (0, 0), fx=0.2, fy=0.2))
#         if cv2.waitKey(0) == 27: return
#     cv2.destroyAllWindows()

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

def contrast_mask(mask : np.ndarray):
    k = 255 / np.max(mask)
    return k * mask

# def compare_masks_red(truth, pred):

#     difference = cv2.subtract(truth, pred)

#     gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
#     _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#     difference[mask != 255] = [0, 0, 255]

#     return difference

# def compare_masks_rgb(truth, pred):

    # max1 = np.amax(truth)
    # max2 = np.amax(pred)

    # height = truth.shape[0]
    # width = truth.shape[1]
    # blank_image = np.zeros((height, width, 3), np.uint8) # BGR
    # blank_image[:,:,1] = np.copy(np.multiply(truth[:,:,0], 255 / max1)) # Green
    # blank_image[:,:,2] = np.copy(np.multiply(pred[:,:,0], 255 / max2)) # Red

    # return blank_image

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

# def visualize_segmentation_result(images, masks, preds=None, figsize=4, nm_img_to_plot=2, n_classes=4, ouput_path=None, epoch=0):

#     cols = 2 if preds is None else 5

#     fig, axes = plt.subplots(nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize))
#     axes[0, 0].set_title("original", fontsize=15) 
#     axes[0, 1].set_title("ground truth", fontsize=15)
#     if not (preds is None):
#         axes[0, 2].set_title("prediction", fontsize=15) 
#         axes[0, 3].set_title("error map", fontsize=15) 
#         axes[0, 4].set_title("overlay", fontsize=15)
    
#     im_id = 0
#     for m in range(0, nm_img_to_plot):
#         axes[m, 0].imshow(images[im_id])
#         axes[m, 0].set_axis_off()
#         axes[m, 1].imshow(colorize_mask(masks[im_id], n_classes=n_classes))
#         axes[m, 1].set_axis_off()        
#         if not (preds is None):
#             axes[m, 2].imshow(colorize_mask(preds[im_id], n_classes=n_classes))
#             axes[m, 2].set_axis_off()
#             axes[m, 3].imshow(visualize_error_mask(create_error_mask(masks[im_id], preds[im_id], num_classes=n_classes)))
#             axes[m, 3].set_axis_off()
#             axes[m, 4].imshow(images[im_id])
#             axes[m, 4].imshow(visualize_error_mask(create_error_mask(masks[im_id], preds[im_id], num_classes=n_classes)), alpha=0.5)
#             axes[m, 4].set_axis_off()
#         im_id += 1

#     if (ouput_path != None):
#         output_name = os.path.join(ouput_path, str(epoch) + '_EPOCH.jpg')
#         fig.savefig(output_name)

#     plt.close()

# def visualize_segmentation_result(images, masks, preds=None, figsize=4, nm_img_to_plot = 1, n_classes=4, output_path=None, epoch=0):

#     cols = 2 if preds is None else 5

#     output_path_name = os.path.join(output_path, str(epoch + 1) + '_EPOCH')
#     try:
#         os.mkdir(output_path_name)
#     except FileExistsError:
#         shutil.rmtree(output_path_name)
#         os.mkdir(output_path_name)

#     for im_id in range(0, nm_img_to_plot):

#         fig, axes = plt.subplots(1, cols, figsize=(cols * figsize, figsize))
#         axes[0].set_title("original", fontsize=15) 
#         axes[1].set_title("ground truth", fontsize=15)
#         if not (preds is None):
#             axes[2].set_title("prediction", fontsize=15) 
#             axes[3].set_title("error map", fontsize=15) 
#             axes[4].set_title("overlay", fontsize=15)

#         axes[0].imshow(images[im_id])
#         axes[0].set_axis_off()
#         axes[1].imshow(colorize_mask(masks[im_id], n_classes=n_classes))
#         axes[1].set_axis_off()        
#         if not (preds is None):
#             axes[2].imshow(colorize_mask(preds[im_id], n_classes=n_classes))
#             axes[2].set_axis_off()
#             axes[3].imshow(visualize_error_mask(create_error_mask(masks[im_id], preds[im_id], num_classes=n_classes)))
#             axes[3].set_axis_off()
#             axes[4].imshow(images[im_id])
#             axes[4].imshow(visualize_error_mask(create_error_mask(masks[im_id], preds[im_id], num_classes=n_classes)), alpha=0.5)
#             axes[4].set_axis_off()

#         if (output_path != None):
#             output_name = os.path.join(output_path_name, str(im_id + 1) + '_image.jpg')
#             fig.savefig(output_name)

#         plt.close()

def visualize_segmentation_result(images, masks, preds=None, figsize=4, nm_img_to_plot = 1, n_classes=4, output_path=None, epoch=0):

    output_path_name = os.path.join(output_path, str(epoch + 1) + '_EPOCH')

    try:
        os.mkdir(output_path_name)
    except FileExistsError:
        shutil.rmtree(output_path_name)
        os.mkdir(output_path_name)

    alpha = 0.6

    for im_id in range(0, nm_img_to_plot):
        k = 255 / np.amax(images[im_id])
        (Image.fromarray((k*images[im_id]).astype(np.uint8))).save(os.path.join(output_path_name, 'image_' + str(im_id + 1) + '_src.png'))
        (Image.fromarray(colorize_mask(masks[im_id], n_classes=n_classes).astype(np.uint8))).save(os.path.join(output_path_name, 'image_' + str(im_id + 1) + '_gt.png'))

        if not (preds is None):
            (Image.fromarray(colorize_mask(preds[im_id], n_classes=n_classes).astype(np.uint8))).save(os.path.join(output_path_name, 'image_' + str(im_id + 1) + '_predicted.png'))
            
            err_mask = create_error_mask(masks[im_id], preds[im_id], num_classes=n_classes)
            (Image.fromarray(visualize_error_mask(err_mask).astype(np.uint8))).save(os.path.join(output_path_name, 'image_' + str(im_id + 1) + '_error.png'))

            (Image.fromarray((alpha*k*images[im_id] + (1 - alpha)*visualize_error_mask(err_mask)).astype(np.uint8))).save(os.path.join(output_path_name, 'image_' + str(im_id + 1) + '_overlay.png'))

def visualize_prediction_result(image, predicted, image_name, figsize=4, output_path=None):

    fig, axes = plt.subplots(1, 3, figsize=(3 * figsize, 1 * figsize))
    axes[0].set_title("Original", fontsize=15)
    axes[1].set_title("Predicted", fontsize=15)
    axes[2].set_title("Overlay", fontsize=15)

    axes[0].imshow(image)
    axes[0].set_axis_off()
    axes[1].imshow(colorize_mask(predicted, n_classes=4))
    axes[1].set_axis_off()
    axes[2].imshow(image)
    axes[2].imshow(colorize_mask(predicted, n_classes=4), alpha=0.5)
    axes[2].set_axis_off()

    if (output_path != None):
        output_name = os.path.join(output_path, image_name)
        fig.savefig(output_name)

    plt.close()


def visualize_line_detection_result(orig_img, edges, hough_orig, hough_edges, num_images, output_path, output_name):

    figsize=15
    fig, axes = plt.subplots(num_images, 4, figsize=(4*figsize, num_images*figsize))
    axes[0,0].set_title("Original", fontsize=15)
    axes[0,1].set_title("Edges", fontsize=15)
    axes[0,2].set_title("Original + lines", fontsize=15)
    axes[0,3].set_title("Edges + lines", fontsize=15)

    img_idx = 0
    for i in range(0, num_images):
        axes[i,0].imshow(orig_img[img_idx])
        axes[i,0].set_axis_off()
        axes[i,1].imshow(edges[img_idx])
        axes[i,1].set_axis_off()
        axes[i,2].imshow(hough_orig[img_idx])
        axes[i,2].set_axis_off()
        axes[i,3].imshow(hough_edges[img_idx])
        axes[i,3].set_axis_off()
        img_idx += 1

    if (output_path != None):
        output_name = os.path.join(output_path, output_name)
        fig.savefig(output_name)

    plt.close()
    
def plot_metrics_history(metrics_values : dict):

    for metric in metrics_values.keys():

        lists = sorted(metrics_values[metric].items())
        x, y = zip(*lists)

        fig = plt.figure(figsize=(12,6))
        plt.plot(x, y, linewidth=3)
        plt.suptitle(metric + ' metric over epochs', fontsize=20)
        plt.ylabel('metric', fontsize=20)
        plt.xlabel('epoch', fontsize=20)
        plt.legend([metric], loc='center right', fontsize=15)
        fig.savefig(os.path.join(metric + '.jpg'))
        plt.show()

def plot_per_class_history(metrics_values : dict):

    epochs = len(metrics_values[0])
    fig = plt.figure(figsize=(12,6))
    i = 0
    for class_num in metrics_values.keys():
        args = [x+1 for x in range(epochs)]
        vals = [y for y in metrics_values[class_num]]
        data = {x : y for x, y in zip(args, vals)}
        lists = sorted(data.items())
        x, y = zip(*lists)
        plt.plot(x, y, linewidth=3)
        i +=1

    plt.suptitle('iou metric per class over epochs', fontsize=20)
    plt.ylabel('metric', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend([j for j in range(i)], loc='center right', fontsize=15)
    fig.savefig(os.path.join('per_class_iou' + '.jpg'))
    plt.show()

def plot_lrs(lrs : list):

    plt.style.use("ggplot")
    fig = plt.figure()
    plt.plot([i + 1 for i in range(0, len(lrs))], lrs)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch #")
    plt.ylabel("Learning Rate")
    fig.savefig(os.path.join('lrs' + '.jpg'))
    plt.close()