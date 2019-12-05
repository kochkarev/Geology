#import plaidml.keras
#plaidml.keras.install_backend()
import numpy as np
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2

def get_imgs_masks(path):
    masks = glob.glob(path)
    imgs = list(map(lambda x: x.replace("_NEW.png", ".jpg"), masks))
    imgs_list = []
    masks_list = []
    for image, mask in zip(imgs, masks):
        imgs_list.append(np.array(Image.open(image)))
        masks_list.append(np.array(Image.open(mask))[...,0]) 

    return imgs_list, masks_list

def get_unmarked_images(path, marked_path):
    marked = glob.glob(os.path.join(marked_path, "*.jpg"))
    all_images = glob.glob(os.path.join(path, "*.jpg"))
    unmarked = set([os.path.basename(img) for img in all_images]) - set([os.path.basename(img) for img in marked])
    return [img for img in all_images if os.path.basename(img) in unmarked]

def get_pairs_from_paths(path):
	images = glob.glob(os.path.join(path,"*.jpg"))
	segmentations = glob.glob(os.path.join(path,"*_NEW.png")) 
	segmentations_d = dict(zip(segmentations,segmentations))
	ret = []
	for im in images:
		seg_bnme = os.path.basename(im).replace(".jpg", "_NEW.png")
		seg = os.path.join(path, seg_bnme)
		assert (seg in segmentations_d),  (im + " is present in " + path + " but " + seg_bnme + " is not found in " + path + " . Make sure annotation image are in .png"  )
		ret.append((im , seg))
	return ret

def resize_imgs_masks(imgs, masks, num_layers=None, patch_size=None):
    
    assert ((num_layers == None) != (patch_size == None)), ("Wrong input params: num_layers or patch_size should be None")

    k = pow(2, num_layers) if num_layers != None else patch_size
    new_imgs = []
    new_masks = []

    for img, mask in zip(imgs, masks):

        assert (img.shape == mask.shape) , ("Mask shape must be equal to image shape")

        height = img.shape[0]
        width = img.shape[1]
        new_height = np.ceil(height / k).astype('int') * k
        new_width = np.ceil(width / k).astype('int') * k

        new_imgs.append(np.pad(img, ((0, new_height - height), (0, new_width - width), (0, 0)), 'constant'))
        new_masks.append(np.pad(mask, ((0, new_height - height), (0, new_width - width), (0, 0)), 'constant'))

    return new_imgs, new_masks

def split_to_patches(img, patch_size, offset, align=None):

    patches = []
    height = img.shape[0]
    width = img.shape[1]
    new_h = height
    new_w = width

    if (img.shape[0] - patch_size) % (patch_size - 2 * offset) != 0:
        new_h = np.ceil((img.shape[0] - patch_size) / (patch_size - 2 * offset)).astype('int') * (patch_size - 2 * offset) + patch_size

    if (img.shape[1] - patch_size) % (patch_size - 2 * offset) != 0:
        new_w = np.ceil((img.shape[1] - patch_size) / (patch_size - 2 * offset)).astype('int') * (patch_size - 2 * offset) + patch_size

    img = np.pad(img, ((0, new_h - height), (0, new_w - width), (0, 0)), 'constant')

    i = 0  
    j = 0
    while (i + patch_size <= img.shape[0]):
        while (j + patch_size <= img.shape[1]):
            patches.append(img[i : i + patch_size, j : j + patch_size, :])
            j += patch_size - 2 * offset
        i += patch_size - 2 * offset
        j = 0
    return patches, (new_h, new_w)

def combine_patches(patches, patch_size, offset, size, orig_size, fill_color = (255,255,255,255)):

    kk = 0
    img = np.full(shape=(size[0], size[1], patches[0].shape[2]), fill_value=fill_color, dtype=patches[0].dtype)
    i = 0
    j = 0
    while (i + patch_size <= size[0]):
        while (j + patch_size <= size[1]):
            img[i+offset : i+patch_size-offset, j+offset : j+patch_size-offset, ...] = patches[kk][offset : patch_size-offset, offset : patch_size-offset, ...]
            j += patch_size - 2 * offset
            kk += 1
        i += patch_size - 2 * offset
        j = 0

    img = img[:orig_size[0], :orig_size[1], ...]
    img[-offset:, ...] = fill_color
    img[:,-offset:,...] = fill_color
    return img