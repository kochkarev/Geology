import numpy as np
import glob
from PIL import Image
import os

def get_imgs_masks(path):
    masks = glob.glob(path)
    imgs = list(map(lambda x: x.replace("_NEW.png", ".jpg"), masks))
    imgs_list = []
    masks_list = []
    for image, mask in zip(imgs, masks):
        imgs_list.append(np.array(Image.open(image)))
        #masks_list.append(np.array(Image.open(mask))[:,:,0])
        masks_list.append(np.array(Image.open(mask)))

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

def resize_imgs_masks(num_layers, imgs, masks):
    k = pow(2, num_layers)
    height = imgs[0].shape[0]
    width = imgs[0].shape[1]

    if (height // k == 0 and width // k == 0):
        return imgs, masks

    new_height = (height // k + 1) * k
    new_width = (width // k + 1) * k

    print("Old height and width: {h} : {w}".format(h=height, w=width))
    print("New height and width: {h} : {w}".format(h=new_height, w=new_width))

    new_imgs = []
    new_masks = []

    for img, mask in zip(imgs, masks):
        new_img = Image.new("RGB", (new_height, new_width))
        new_img.paste(Image.fromarray(img))
        new_mask = Image.new("RGB", (new_height, new_width))
        new_mask.paste(Image.fromarray(mask))

        new_imgs.append(np.array(new_img))
        new_masks.append(np.array(new_mask)[:,:,0])

    return new_imgs, new_masks