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
        
    return np.asarray(imgs_list, dtype=np.float32) / 255, np.asarray(masks_list, dtype=np.float32)