import numpy as np
import glob
from PIL import Image

def get_imgs_masks(path):
    masks = glob.glob(path)
    imgs = list(map(lambda x: x.replace("_NEW.png", ".jpg"), masks))
    imgs_list = []
    masks_list = []
    for image, mask in zip(imgs, masks):
        imgs_list.append(np.array(Image.open(image).resize((384,384))))
        masks_list.append(np.array(Image.open(mask).resize((384,384)))[:,:,0])

    #return np.asarray(imgs_list, dtype=np.float32) / 255, np.asarray(masks_list, dtype=np.float32)
    return imgs_list, masks_list

def resize_imgs_masks(num_layers, imgs, masks):
    k = pow(2, num_layers)
    height = imgs[0].shape[0]
    width = imgs[0].shape[1]

    if (height // k == 0 and width // k == 0):
        return imgs, masks

    new_height = (height // k + 1) * k
    new_width = (width // k + 1) * k

    new_imgs = []
    new_masks = []

    for img, mask in zip(imgs, masks):
        new_img = Image.new("RGB", (new_height, new_width))
        new_img.paste(img)
        new_mask = Image.new("RGB", (new_height, new_width))
        new_mask.paste(mask)

        new_imgs.append(np.array(new_img))
        new_masks.append(np.array(new_mask))

    return new_imgs, new_masks