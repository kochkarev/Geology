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
        #imgs_list.append(np.array(Image.open(image)))
        imgs_list.append(cv2.imread(image))
        #masks_list.append(np.array(Image.open(mask))) 
        masks_list.append(cv2.imread(mask))

    return imgs_list, masks_list

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

def make_patches(img, size=256):
    
    assert (img.ndim == 3), ("Wrong input params: 3-channel input expected")
    assert (img.shape[0] % size == 0 and img.shape[1] % size == 0), ("Wrong input params: dimensions of image mod size must be zero")

    patches_list = []
   
    for i in range(img.shape[0] // size):
        for j in range(img.shape[1] // size):
            patches_list.append(img[i*size : i*size+size, j*size : j*size+size, :])

    return np.stack(patches_list) 

def reconstruct_from_patches(img, org_img_size, size=256):

    assert (type(org_img_size) is tuple), ("Wrong input params: size must be a tuple")
    assert (img.ndim == 4), ("Wrong input params: 4-dim array expected")
    
    i_max = (org_img_size[0] // size)
    j_max = (org_img_size[1] // size)
       
    kk = 0
    img_bg = np.zeros((org_img_size[0], org_img_size[1], 3), dtype=img[0].dtype)
    
    for i in range(i_max):
        for j in range(j_max):
            img_bg[i*size:i*size+size, j*size:j*size+size, :] = img[kk, :, :, :]
            kk += 1        

    return img_bg

def generate_patches_list(imgs, masks, patch_size):

    new_images = []
    new_masks = []
    
    for img, mask in zip(imgs, masks):

        new_images.append(make_patches(img, patch_size))
        new_masks.append(make_patches(mask, patch_size))

    return new_images, new_masks

def convert_patches_list(imgs, masks):

    assert (imgs[0].shape[0] == masks[0].shape[0]), ("Wrong input params: # of images and masks must match")

    new_images = []
    new_masks = []

    for img, mask in zip(imgs, masks):

        for i in range(img.shape[0]):
            new_images.append(img[i, :, :, :])
            new_masks.append(mask[i, :, :, :])
    
    return new_images, new_masks