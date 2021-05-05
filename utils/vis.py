from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def hex_to_rgb(hex: str):
    h = hex.lstrip('#')
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def to_heat_map(img, name='jet'):
    assert img.ndim == 2, 'shape {} is unsupported'.format(img.shape)
    img_min, img_max = np.min(img), np.max(img)
    assert (img_min >= 0.0 and img_max <= 1.0), f'invalid range {img_min} - {img_max}'
    img = img / img_max if img_max != 0 else img
    cmap = plt.get_cmap(name)
    heat_img = cmap(img)[..., 0:3]
    return (heat_img * 255).astype(np.uint8)


def _fill_offset(arr, offset: int, value: int = 0):
    if offset > 0:
        arr[:offset, :, ...] = value
        arr[-offset:, :, ...] = value
        arr[:, :offset, ...] = value
        arr[:, -offset:, ...] = value
    return arr


def colorize_mask(mask: np.ndarray, offset: int, codes_to_colors: Dict) -> np.ndarray:
    assert mask.ndim == 2, 'only 2d masks are supported'
    colorized = np.zeros(mask.shape + (3,), dtype=np.uint8)
    codes_to_colors_rgb = {code: hex_to_rgb(color) for code, color in codes_to_colors.items()}
    for code, color in codes_to_colors_rgb.items():
        colorized[mask == code, :] = color
    colorized = _fill_offset(colorized, offset)
    return colorized


def error_mask(img: np.ndarray, pred: np.ndarray, offset: int = 0) -> np.ndarray:
    assert img.ndim == 2 and pred.ndim == 2
    assert img.shape == pred.shape, f'Expected gt and mask with same shape, got {img.shape} and {pred.shape}'
    error_map = img != pred
    error_map = _fill_offset(error_map, offset)
    return error_map


def colorize_error_mask(mask: np.ndarray, color_correct=(0, 255, 0), color_error=(255, 0, 0), offset: int = 0):
    assert mask.ndim == 2, 'Expected (H x W) output'
    colorized = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for ch in range(3):
        colorized[:, :, ch] = np.where(mask == 1, color_error[ch], color_correct[ch])
    colorized = _fill_offset(colorized, offset)
    return colorized


def vis_segmentation(image: np.ndarray, mask: np.ndarray, pred: np.ndarray, offset: int, codes_to_colors: Dict,
                     out_folder: Path, name: str, alpha=0.75):
    # print(f'image: {np.min(image), np.max(image), np.sum(image), pred.dtype}')
    # print(f'pred: {np.min(pred), np.max(pred), np.sum(pred), pred.dtype}')
    # print(f'mask: {np.min(mask), np.max(mask), np.sum(mask), pred.dtype}')
    mask_colorized = colorize_mask(pred, offset=offset, codes_to_colors=codes_to_colors) 
    Image.fromarray(mask_colorized).save(out_folder / f'{name}_pred.jpg')            
    err_mask = error_mask(mask, pred, offset=offset)
    err_vis = colorize_error_mask(err_mask, offset=offset)
    Image.fromarray(err_vis).save(out_folder / f'{name}_error.jpg')
    overlay = (alpha * image + (1 - alpha) * err_vis).astype(np.uint8)
    Image.fromarray(overlay).save(out_folder / f'{name}_overlay.jpg')


def plot_lrs(lrs: list, output_path: Path):
    plt.style.use("ggplot")
    fig = plt.figure()
    plt.plot([i + 1 for i in range(0, len(lrs))], lrs)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch #")
    plt.ylabel("Learning Rate")
    fig.savefig(output_path / 'lrs.jpg')
    plt.close()
