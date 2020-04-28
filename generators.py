import numpy as np
from random import randrange, randint
from skimage.transform import resize
from math import sqrt
from scipy import ndimage
import os
from config import classes_mask
import json
from time import time
from PIL import Image
from utils import colorize_mask
from tensorflow.keras.utils import to_categorical

class PatchGenerator:

    def __init__(self, images: np.ndarray, masks: np.ndarray, names: str, patch_size: int, batch_size: int, full_augment: bool = False):

        self.images = images
        self.masks = masks
        self.names = names
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.full_augment = full_augment
        self.stat = {classes_mask[i] : 0 for i in classes_mask}
        self.load_heatmaps()

    def load_heatmaps(self):
        print('Loading heatmaps..')
        t1 = time()
        classes = [cl for cl in classes_mask.values()]
        self.heatmaps = []
        for input_img in self.names:
            self.heatmaps.append([(np.load((os.path.join("input", "heatmaps", cl + "__" + input_img.replace(".jpg", ".npz")))))["arr_0"] for cl in classes])
        t2 = time()
        print(f'Heatmaps loaded in {t2-t1} seconds')

    def __iter__(self):
        
        assert (self.images.shape[:3] == self.masks.shape[:3]), ("Original images and masks must be equal shape")

        classes = {cl : i for i, cl in enumerate(classes_mask.values())}
        aa = np.arange(self.masks[0].shape[0] * self.masks[0].shape[1])
        a=np.indices(self.masks[0].shape)
        _xx=np.ndarray.flatten(a[0])
        _yy=np.ndarray.flatten(a[1])

        with open(os.path.join("input", "ores.json")) as ores_json:
            ores = json.load(ores_json)

        while(True):
            
            cur_class = min(self.stat.keys(), key=(lambda k: self.stat[k]))
            # print(cur_class)
            pp = np.fromiter(ores[cur_class].values(), dtype=np.float64)
            pp /= np.sum(pp)
            batch_files = np.random.choice(a=self.images.shape[0], size=self.batch_size, p=pp)

            batch_x = []
            batch_y = []

            for batch_idx in batch_files:

                if self.full_augment:
                    etha_max = 2
                    etha = np.random.uniform(1 / etha_max, etha_max)
                    angle = np.random.random_integers(0, 360)
                    new_size = int(np.ceil(self.patch_size * sqrt(2) * etha))

                p_s = self.heatmaps[batch_idx]
                
                p = p_s[classes[cur_class]]
                if self.full_augment:
                    while (True):
                        n = np.random.choice(a=aa, p=np.ndarray.flatten(p))
                        if (_xx[n] + new_size < self.masks[batch_idx].shape[0] and _yy[n] + new_size < self.masks[batch_idx].shape[1]):
                            break
                else:
                    while (True):
                        n = np.random.choice(a=aa, p=np.ndarray.flatten(p))
                        if (_xx[n] + self.patch_size < self.masks[batch_idx].shape[0] and _yy[n] + self.patch_size < self.masks[batch_idx].shape[1]):
                            break

                # Choosing patch
                if self.full_augment:
                    yy = self.masks[batch_idx, _xx[n] : _xx[n] + new_size, _yy[n] : _yy[n] + new_size, :]
                    xx = self.images[batch_idx, _xx[n] : _xx[n] + new_size, _yy[n] : _yy[n] + new_size, :]
                else:
                    yy = self.masks[batch_idx, _xx[n] : _xx[n] + self.patch_size, _yy[n] : _yy[n] + self.patch_size, :]
                    xx = self.images[batch_idx, _xx[n] : _xx[n] + self.patch_size, _yy[n] : _yy[n] + self.patch_size, :]
                
                xx = Image.fromarray((255*xx).astype(np.uint8))
                yy = Image.fromarray((np.argmax(yy, axis=2)).astype(np.uint8))

                if self.full_augment:
                    # Rotate
                    xx = xx.rotate(angle=angle, resample=Image.BICUBIC)
                    yy = yy.rotate(angle=angle, resample=Image.NEAREST)
                    
                    # Rescale
                    new_size1 = int(np.ceil(self.patch_size * etha))
                    ii = abs(new_size - new_size1) // 2
                    xx = xx.crop((ii, ii, ii + new_size1, ii + new_size1))
                    yy = yy.crop((ii, ii, ii + new_size1, ii + new_size1))

                    xx = xx.resize(size=(self.patch_size, self.patch_size), resample=Image.BICUBIC)
                    yy = yy.resize(size=(self.patch_size, self.patch_size), resample=Image.NEAREST)

                # Flip 
                if randint(0, 1) == 0:
                    xx, yy = xx.transpose(Image.FLIP_TOP_BOTTOM), yy.transpose(Image.FLIP_TOP_BOTTOM)
                else:
                    xx, yy = xx.transpose(Image.FLIP_LEFT_RIGHT), yy.transpose(Image.FLIP_LEFT_RIGHT)

                # Rotate 90
                rotate_flg = randint(0, 3)
                if rotate_flg == 0:
                    xx, yy = xx.transpose(Image.ROTATE_90), yy.transpose(Image.ROTATE_90)
                elif rotate_flg == 1:
                    xx, yy = xx.transpose(Image.ROTATE_180), yy.transpose(Image.ROTATE_180)
                elif rotate_flg == 2:
                    xx, yy = xx.transpose(Image.ROTATE_270), yy.transpose(Image.ROTATE_270)

                yy = to_categorical(np.array(yy).astype(np.uint8), num_classes=len(classes_mask.keys()))
                batch_x.append(np.array(xx).astype(np.float32) / 255)
                batch_y.append(yy)

                unique, counts = np.unique(np.argmax(yy, axis=2), return_counts=True)
                aaa = dict(zip(unique, counts))
                for class_n in range(len(classes_mask)):
                    if class_n in aaa:
                        self.stat[classes_mask[class_n]] += aaa[class_n]

            batch_x = np.stack(batch_x)
            batch_y = np.stack(batch_y)

            yield (batch_x, batch_y)