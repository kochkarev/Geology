import numpy as np
from random import randrange, randint
from skimage.transform import resize
from math import sqrt
from scipy import ndimage
import os
from config import classes_mask
import json

class PatchGenerator:

    def __init__(self, images: np.ndarray, masks: np.ndarray, names: str, patch_size: int, batch_size: int, augment: bool = True):

        self.images = images
        self.masks = masks
        self.names = names
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.augment = augment
        self.stat = {classes_mask[i] : 0 for i in classes_mask}
        self.load_heatmaps()

    def load_heatmaps(self):
        classes = [cl for cl in classes_mask.values()]
        self.heatmaps = []
        for input_img in self.names:
            self.heatmaps.append([(np.load((os.path.join("input", "heatmaps", cl + "__" + input_img.replace(".jpg", ".npz")))))["arr_0"] for cl in classes])

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
            pp = np.fromiter(ores[cur_class].values(), dtype=np.float64)
            pp /= np.sum(pp)
            batch_files = np.random.choice(a=self.images.shape[0], size=self.batch_size, p=pp)

            batch_x = []
            batch_y = []

            for batch_idx in batch_files:

                etha_max = 2
                etha = np.random.uniform(1 / etha_max, etha_max)
                angle = np.random.random_integers(0, 360)
                new_size = int(np.ceil(self.patch_size * sqrt(2) * etha))

                p_s = self.heatmaps[batch_idx]
                
                p = p_s[classes[cur_class]]
                    
                while (True):
                    n = np.random.choice(a=aa, p=np.ndarray.flatten(p))
                    if (_xx[n] + new_size < self.masks[batch_idx].shape[0] and _yy[n] + new_size < self.masks[batch_idx].shape[1]):
                        break

                yy = self.masks[batch_idx, _xx[n] : _xx[n] + new_size, _yy[n] : _yy[n] + new_size, :]
                xx = self.images[batch_idx, _xx[n] : _xx[n] + new_size, _yy[n] : _yy[n] + new_size, :]

                # Rotating
                xx, yy = ndimage.rotate(xx, angle, reshape=False), ndimage.rotate(yy, angle, reshape=False)

                new_size1 = int(np.ceil(self.patch_size * etha))
                ii = abs(new_size - new_size1) // 2
                xx, yy = xx[ii : ii + new_size1, ii : ii + new_size1, :], yy[ii : ii + new_size1, ii : ii + new_size1, :]

                # Scaling
                xx, yy = resize(xx, (self.patch_size, self.patch_size)), resize(yy, (self.patch_size, self.patch_size))

                # Mirroring
                if randint(0, 1) == 0:
                    xx, yy = np.flipud(xx), np.flipud(yy)
                else:
                    xx, yy = np.fliplr(xx), np.fliplr(yy)

                # Rotating 90
                if randint(0, 1) == 0:
                    xx, yy = np.rot90(xx), np.rot90(yy)

                batch_x.append(xx)
                batch_y.append(yy)

                yyy = np.argmax(yy, axis=2)
                unique, counts = np.unique(yyy, return_counts=True)
                aaa = dict(zip(unique, counts))
                for class_n in range(len(classes_mask)):
                    if class_n in aaa:
                        self.stat[classes_mask[class_n]] += aaa[class_n]

            batch_x = np.stack(batch_x)
            batch_y = np.stack(batch_y)

            yield (batch_x, batch_y)