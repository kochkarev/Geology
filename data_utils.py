import numpy as np
import glob
from PIL import Image
from os import listdir
from os.path import join, isfile, basename
import json
import skimage.io as io
from config import classes_mask

# depricated
# def _get_imgs_masks(path):
#     masks = glob.glob(path)
#     imgs = list(map(lambda x: x.replace("_NEW.png", ".jpg"), masks))
#     imgs_list = []
#     masks_list = []
#     for image, mask in zip(imgs, masks):
#         imgs_list.append(np.array(Image.open(image)))
#         masks_list.append(np.array(Image.open(mask))[...,0]) 

#     return imgs_list, masks_list

def get_imgs_masks(path: str, load_test: bool = True, return_names: bool = False):

    train_imgs_path = join(path, "imgs", "train")
    train_masks_path = join(path, "masks", "train")
    test_imgs_path = join(path, "imgs", "test")
    test_masks_path = join(path, "masks", "test")

    train_imgs_names = [f for f in listdir(train_imgs_path) if isfile(join(train_imgs_path, f))]
    train_masks_names = [f for f in listdir(train_masks_path) if isfile(join(train_masks_path, f))]
    test_imgs_names = [f for f in listdir(test_imgs_path) if isfile(join(test_imgs_path, f))]
    test_masks_names = [f for f in listdir(test_masks_path) if isfile(join(test_masks_path, f))]

    n_train, n_test = len(train_imgs_names), len(test_imgs_names)
    img_shape = np.array(Image.open(join(train_imgs_path, train_imgs_names[0]))).shape

    train_data = np.zeros([n_train, img_shape[0], img_shape[1], 3], dtype=np.float32)
    train_masks = np.zeros([n_train, img_shape[0], img_shape[1]], dtype=np.uint8)
    if load_test:
        test_data = np.zeros([n_test, img_shape[0], img_shape[1], 3], dtype=np.float32)
        test_masks = np.zeros([n_test, img_shape[0], img_shape[1]], dtype=np.uint8)
    else:
        test_data = None
        test_masks = None

    for i, name in enumerate(train_imgs_names):
        train_data[i, ...] = np.array(Image.open(join(train_imgs_path, name))).astype(np.float32) / 255
    for i, name in enumerate(train_masks_names):
        train_masks[i, ...] = np.array(Image.open(join(train_masks_path, name)))[..., 0]
    if load_test:
        for i, name in enumerate(test_imgs_names):
            test_data[i, ...] = np.array(Image.open(join(test_imgs_path, name))).astype(np.float32) / 255
        for i, name in enumerate(test_masks_names):
            test_masks[i, ...] = np.array(Image.open(join(test_masks_path, name)))[..., 0]
    
    return (train_data, test_data, train_masks, test_masks) if (not return_names) else \
        (train_data, test_data, train_masks, test_masks, train_imgs_names, test_imgs_names)

    

def get_unmarked_images(path, marked_path):
    marked = glob.glob(join(marked_path, "*.jpg"))
    all_images = glob.glob(join(path, "*.jpg"))
    unmarked = set([basename(img) for img in all_images]) - set([basename(img) for img in marked])
    return [img for img in all_images if basename(img) in unmarked]

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

def generate_lists_mineral(input_path: str, output_path: str):

    classes = [cl for cl in classes_mask.values()]
    with open(os.path.join("input", "dataset.json")) as dataset_json:
        names = json.load(dataset_json)
    marked_images = names["BoxA_DS1"]["train"]
    stat = {cl : {} for cl in classes}

    for img in marked_images:

        mask = np.array(Image.open(os.path.join(input_path, img.replace(".jpg","_NEW.png"))))[:,:,0]

        unique, counts = np.unique(mask, return_counts=True)
        aaa = dict(zip(unique, counts))

        for class_n in range(len(classes_mask)):
            if class_n in aaa:
                stat[classes_mask[class_n]][img] = str(aaa[class_n])
            else:
                stat[classes_mask[class_n]][img] = "0"

    with open(output_path, 'w') as fp:
        json.dump(stat, fp, indent=4)