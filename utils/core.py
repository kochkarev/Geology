import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from config import classes_colors, classes_mask
from tensorflow.keras.utils import to_categorical
from pathlib import Path

def plot_segm_history(history, output_path, metrics=['iou'], losses=['loss']):
    # summarize history for iou
    fig1 = plt.figure(figsize=(12,6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle('metrics over epochs', fontsize=20)
    plt.ylabel('metric', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend(metrics, loc='center right', fontsize=15)
    fig1.savefig(os.path.join(output_path, 'metrics.jpg'))
    # summarize history for loss
    fig2 = plt.figure(figsize=(12,6))    
    for loss in losses:
        plt.plot(history.history[loss], linewidth=3)
    plt.suptitle('loss over epochs', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend(losses, loc='center right', fontsize=15)
    fig2.savefig(os.path.join(output_path, 'loss.jpg'))

def hex_to_rgb(hex: str):
    h = hex.lstrip('#')
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

def to_heat_map(img, name='jet'):
    assert img.ndim == 2, 'shape {} is unsupported'.format(img.shape)
    assert (np.min(img) >= 0.0) and (np.max(img) <= 1.0), 'invalid range {} - {}'.format(np.min(img), np.max(img))
    cmap = plt.get_cmap(name)
    heat_img = cmap(img)[..., 0:3]
    return (heat_img * 255).astype(np.uint8)


def colorize_mask(mask, n_classes):
    colorized = np.zeros_like(mask)
    colors = [hex_to_rgb(color) for color in classes_colors]
    for c in range(n_classes):
        if mask.ndim == 3:
            colorized[:,:,0] += ((mask[:,:,0] == c) * (colors[c][0])).astype('uint8')
            colorized[:,:,1] += ((mask[:,:,0] == c) * (colors[c][1])).astype('uint8')
            colorized[:,:,2] += ((mask[:,:,0] == c) * (colors[c][2])).astype('uint8')
        elif mask.ndim == 2:
            colorized += ((mask == c) * (colors[c][2])).astype('uint8')
    return colorized

def create_error_mask(img : np.ndarray, pred : np.ndarray, num_classes : int = 4):

    assert img.ndim == 2 and pred.ndim == 2, 'Expected (H x W) masks, got img with shape {} and pred with shape {}'.format(img.shape, pred.shape)
    assert img.shape == pred.shape, f'Expected gt and mask with same shape, got {img.shape} and {pred.shape}'

    masks = []
    for i in range(num_classes):
        masks.append(np.array(np.where(img == i, 1, 0) * np.where(pred == i, 1, 0), dtype=np.uint8))
    res = np.zeros_like(img, dtype=np.uint8)
    for mask in masks:
        res += mask

    return res

def visualize_error_mask(mask : np.ndarray, show=False):

    assert (mask.ndim == 2), ('Expected (H x W) output')
    assert ((np.array([0,1])==np.unique(mask)).all() == True), ('Expected binary mask')

    mask = np.asarray(np.dstack((mask, mask, mask)), dtype=np.uint8)
    mask = np.array(np.where(mask == (0,0,0), (255,0,0), (0,255,0)), dtype=np.uint8)

    if show:
        Image.fromarray(mask).show()

    return mask

def error_per_class(gt: np.ndarray, pred: np.ndarray, n_classes: int):

    gt, pred = to_categorical(gt, n_classes), to_categorical(pred, n_classes)
    results = []
    for i in range(n_classes):
        mask = np.where(gt[...,i] == 1, 1, 0) * np.where(pred[...,i] == 1, 1, 0)
        results.append(np.sum(mask) / (np.sum(gt[...,i])))
    return results

def visualize_segmentation_result(images, masks, preds, names, n_classes, output_path: Path, epoch):
    output_path_name = output_path / f'epoch_{epoch+1}'
    err_log_name = output_path_name / "err_log.txt"
    if os.path.exists(err_log_name):
        os.remove(err_log_name)
    err_log = open(err_log_name, "a+")
    alpha = 0.75

    for i in range(images.shape[0]):
        if preds is not None:
            Image.fromarray(colorize_mask(np.dstack((preds[i],preds[i],preds[i])), n_classes=n_classes).astype(np.uint8)).save(
                output_path_name / f'image_{i + 1}_pred.jpg'
            )            
            err_mask = create_error_mask(masks[i], preds[i], num_classes=n_classes)

            err_per = error_per_class(masks[i], preds[i], n_classes)
            err_log.write(f'image_{i + 1} ({100*(np.sum(err_mask) / (err_mask.shape[0]*err_mask.shape[1])):.3f}%):\n')
            for j in range(n_classes):
                err_log.write(f'    {classes_mask[j]}: {err_per[j] * 100:.3f}%\n')

            err_vis = visualize_error_mask(err_mask)
            Image.fromarray(err_vis.astype(np.uint8)).save(
                output_path_name / f'image_{i + 1}_error.jpg'
            )
            Image.fromarray((alpha*images[i] + (1 - alpha)*err_vis).astype(np.uint8)).save(
                output_path_name / f'image_{i + 1}_overlay.jpg'
            )

def visualize_prediction_result(image, predicted, image_name, figsize=4, output_path=None):

    fig, axes = plt.subplots(1, 3, figsize=(3 * figsize, 1 * figsize))
    axes[0].set_title("Original", fontsize=15)
    axes[1].set_title("Predicted", fontsize=15)
    axes[2].set_title("Overlay", fontsize=15)

    axes[0].imshow(image)
    axes[0].set_axis_off()
    axes[1].imshow(colorize_mask(np.dstack((predicted, predicted, predicted)), n_classes=4))
    axes[1].set_axis_off()
    axes[2].imshow(image)
    axes[2].imshow(colorize_mask(np.dstack((predicted, predicted, predicted)), n_classes=4), alpha=0.5)
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
    
# def plot_metrics_history(metrics_values: dict, output_path: Path):
#     for metric in metrics_values.keys():
#         lists = sorted(metrics_values[metric].items())
#         x, y = zip(*lists)
#         fig = plt.figure(figsize=(12,6))
#         plt.plot(x, y, linewidth=3)
#         plt.suptitle(metric + ' metric over epochs', fontsize=20)
#         plt.ylabel('metric', fontsize=20)
#         plt.xlabel('epoch', fontsize=20)
#         plt.legend([metric], loc='center right', fontsize=15)
#         fig.savefig(output_path / (metric + '.jpg'))

# def plot_per_class_history(metrics_values: dict, output_path: Path):
#     epochs = len(metrics_values[0])
#     fig = plt.figure(figsize=(12,6))
#     i = 0
#     for cl in metrics_values.keys():
#         args = [x+1 for x in range(epochs)]
#         vals = [y for y in metrics_values[cl]]
#         data = {x : y for x, y in zip(args, vals)}
#         lists = sorted(data.items())
#         x, y = zip(*lists)
#         plt.plot(x, y, linewidth=3, color=classes_colors[cl])
#         i +=1
#     plt.suptitle('metric per class over epochs', fontsize=20)
#     plt.ylabel('metric', fontsize=20)
#     plt.xlabel('epoch', fontsize=20)
#     plt.legend([classes_mask[j] for j in range(i)], loc='center right', fontsize=15)
#     fig.savefig(output_path / 'per_class_metric.jpg')

def plot_metrics(metrics, metric_name, output_path: Path):
    epochs = len(metrics)
    n_classes = len(metrics[0]) - 1
    
    # --- per class metric ---
    fig = plt.figure(figsize=(12,6))
    # ax = plt.axes()
    # ax.set_facecolor('white')
    for cl in range(n_classes):
        x = [x+1 for x in range(epochs)]
        y = [metrics[i][cl] for i in range(epochs)]
        plt.plot(x, y, color=classes_colors[cl])
    # plt.suptitle(f'{metric_name} per class over epochs', fontsize=20)
    plt.ylabel(f'{metric_name}', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend([classes_mask[j] for j in range(n_classes)], loc='center right', fontsize=15)
    fig.savefig(output_path / f'{metric_name}_per_class.png')
    
    # --- all class metric ---
    fig = plt.figure(figsize=(12,6))
    # ax = plt.axes()
    # ax.set_facecolor('white')
    x = [x+1 for x in range(epochs)]
    y = [metrics[i][-1] for i in range(epochs)]
    plt.plot(x, y)
    # plt.suptitle(f'{metric_name} over epochs', fontsize=20)
    plt.ylabel(f'{metric_name}', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    fig.savefig(output_path / f'{metric_name}.png')


def plot_lrs(lrs: list, output_path: Path):

    plt.style.use("ggplot")
    fig = plt.figure()
    plt.plot([i + 1 for i in range(0, len(lrs))], lrs)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch #")
    plt.ylabel("Learning Rate")
    fig.savefig(output_path / 'lrs.jpg')
    plt.close()
