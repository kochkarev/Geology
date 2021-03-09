import numpy as np


def split_to_patches(img: np.ndarray, patch_size: int, offset: int, overlay):

    assert overlay == 0 or overlay == 0.5 or overlay == 0.25, 'Overlay should be 100, 50 or 25%'
    
    overlay = 1 - overlay
    eff_size = patch_size - 2 * offset
    shift = int(overlay * eff_size)
    patches = []

    height, width = img.shape[:2]
    new_h, new_w = height, width
    if (img.shape[0] - patch_size) % shift != 0:
        new_h = int(np.ceil((img.shape[0] - patch_size) / shift)) * shift + patch_size

    if (img.shape[1] - patch_size) % shift != 0:
        new_w = int(np.ceil((img.shape[1] - patch_size) / shift)) * shift + patch_size

    img = np.pad(img, ((0, new_h - height), (0, new_w - width), (0, 0)), 'constant')
    i, j = 0, 0
    while (i + patch_size <= img.shape[0]):
        while (j + patch_size <= img.shape[1]):
            patches.append(img[i : i + patch_size, j : j + patch_size, :])
            j += shift
        i, j = i + shift, 0
    
    return patches

def combine_patches(patches, patch_size, offset, overlay, orig_shape):

    assert (overlay == 0 or overlay == 0.5 or overlay == 0.25), ('Overlay should be 100, 50 or 25%')
    
    overlay = 1 - overlay
    eff_size = patch_size - 2 * offset
    shift = int(overlay * eff_size)
    height, width = orig_shape[:2]
    new_h, new_w = height, width
    if (orig_shape[0] - patch_size) % shift != 0:
        new_h = int(np.ceil((orig_shape[0] - patch_size) / shift)) * shift + patch_size

    if (orig_shape[1] - patch_size) % shift != 0:
        new_w = int(np.ceil((orig_shape[1] - patch_size) / shift)) * shift + patch_size

    # result = np.zeros(shape=(new_h, new_w, orig_shape[2]), dtype=patches[0].dtype)
    result = np.zeros(shape=(new_h, new_w, orig_shape[2]), dtype=np.float32)
    weights = np.zeros(shape=(new_h, new_w, orig_shape[2]), dtype=np.uint8)
    i, j, k = 0, 0, 0
    while (i + patch_size <= new_h):
        while (j + patch_size <= new_w):
            result[i + offset : i + patch_size - offset, j + offset : j + patch_size - offset] += patches[k][offset : patch_size - offset, offset : patch_size - offset, ...]
            weights[i + offset : i + patch_size - offset, j + offset : j + patch_size - offset] += 1
            j, k = j + shift, k + 1
        i, j = i + shift, 0
    weights[weights == 0] = 1
    result /= weights
    result = result[:orig_shape[0], :orig_shape[1], ...] # Crop empty border on right and bottom
    return result[offset : -offset, offset : -offset, ...] # Crop convolution offset
