import numpy as np
from random import randrange, randint
from skimage.transform import resize
from math import sqrt
from scipy import ndimage

class PatchGenerator:

    def __init__(self, images : np.ndarray, masks : np.ndarray, patch_size : int, batch_size : int, augment : bool = True):

        self.images = images
        self.masks = masks
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.augment = augment

    def __iter__(self):

        while(True):
        
            assert (self.images.shape[:3] == self.masks.shape[:3]), ("Original images and masks must be equal shape")

            batch_files = np.random.choice(a=self.images.shape[0], size=self.batch_size)
            etha_max = 2

            batch_x = []
            batch_y = []

            for batch_idx in batch_files:

                etha_max = 2
                etha = np.random.uniform(1 / etha_max, etha_max)
                angle = np.random.random_integers(0, 360)
                new_size = int(np.ceil(self.patch_size * sqrt(2) * etha))

                i = np.random.choice(a=self.images.shape[1]-new_size-1)
                j = np.random.choice(a=self.images.shape[2]-new_size-1)
                xx = self.images[batch_idx, i : i+self.patch_size, j : j+self.patch_size, :]
                yy = self.masks[batch_idx, i : i+self.patch_size, j : j+self.patch_size, :]

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

            batch_x = np.stack(batch_x)
            batch_y = np.stack(batch_y)

            yield (batch_x, batch_y)