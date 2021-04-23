import math
from typing import Iterable, List, Tuple, Union

import numpy as np
from tqdm import tqdm


def _get_patch_coords(img_shape: Tuple[int], patch_size: int, conv_offset: int, overlay: int):
    h, w = img_shape[:2]
    pps = patch_size - 2 * conv_offset
    s = pps - overlay
    nh = math.ceil((h - 2 * conv_offset) / s)
    nw = math.ceil((w - 2 * conv_offset) / s)
    coords = []
    for i in range(nh):
        y = min(i * s, h - patch_size)
        for j in range(nw):
            x = min(j * s, w - patch_size)
            coords.append((y, x))
    return coords


def split_into_patches(img: np.ndarray, patch_size: int, conv_offset: int, overlay: Union[int, float]) -> List[np.ndarray]:
    """
    Splits image (>= 2 dimensions) into patches.

    Args:
        img (np.ndarray): source image
        patch_size (int): patch size in pixels
        conv_offset (int): conv offset in pixels
        overlay (Union[int, float]): either float in [0, 1] (fraction of patch size) or int in pixels

    Returns:
        List[np.ndarray]: list of extracted patches
    """
    assert img.ndim >= 2
    if isinstance (overlay, float):
        overlay = int(patch_size * overlay)
    coords = _get_patch_coords(img.shape, patch_size, conv_offset, overlay)
    patches = []
    for coord in coords:
        y, x = coord
        patch = img[y : y + patch_size, x : x + patch_size, ...]
        patches.append(patch)
    return patches


def combine_from_patches(patches: Iterable[np.ndarray], patch_s: int, conv_offset: int, overlay: Union[int, float],
                         src_size: Tuple[int, int], border_fill_val=0) -> np.ndarray:
    """
    Combines patches back into image.

    Args:
        patches (Iterable[np.ndarray]): patches
        patch_s (int): patch size in pixels
        conv_offset (int): conv offset in pixels
        overlay (Union[int, float]): either float in [0, 1] (fraction of patch size) or int in pixels
        src_size (Tuple[int, int]): target image shape
        border_fill_val (int, optional): value to fill the conv offset border. Defaults to 0.

    Returns:
        np.ndarray: combined image
    """
    if isinstance (overlay, float):
        overlay = int(patches[0].shape[0] * overlay)
    h, w = src_size[:2]
    target_shape = (h, w) + patches[0].shape[2:]
    img = np.zeros(target_shape, dtype=np.float) + border_fill_val
    density = np.zeros_like(img)
    coords = _get_patch_coords(img.shape, patch_s, conv_offset, overlay)
    for i, coord in enumerate(coords):
        y, x = coord
        y0, y1 = y + conv_offset, y + patch_s - conv_offset
        x0, x1 = x + conv_offset, x + patch_s - conv_offset
        img[y0: y1, x0: x1, ...] += patches[i][conv_offset: patch_s - conv_offset, conv_offset: patch_s - conv_offset, ...]
        density[y0: y1, x0: x1, ...] += 1
    density[density == 0] = 1
    img /= density
    img = img.astype(patches[0].dtype)
    return img


def test_spit_combine_random(n_tests=100, eps=1e-3):
    for _ in tqdm(range(n_tests)):
        h = np.random.randint(100, 5000)
        w = np.random.randint(100, 5000)
        patch_s = np.random.randint(16, 1024)
        patch_s = min(h, w, patch_s)
        img = np.random.random((h, w))
        conv_offset = min(np.random.randint(50), patch_s // 4)
        overlay = np.random.randint(0, patch_s // 2)
        patches = split_into_patches(img, patch_s, conv_offset, overlay)
        img_reconstructed = combine_from_patches(patches, patch_s, conv_offset, overlay, img.shape)
        img_crop = img[conv_offset:-conv_offset, conv_offset:-conv_offset]
        img_reconstructed_crop = img_reconstructed[conv_offset:-conv_offset, conv_offset:-conv_offset]
        assert np.sum(np.abs(img_crop - img_reconstructed_crop)) < eps
    print('ok')

# test_spit_combine_random()
