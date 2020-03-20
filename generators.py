import numpy as np
from random import randrange
from skimage.transform import resize

class PatchGenerator:

    def __init__(self, images : np.ndarray, masks : np.ndarray, patch_size : int, batch_size : int, augment : bool = False):

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

                i = np.random.choice(a=self.images.shape[1]-self.patch_size-1)
                j = np.random.choice(a=self.images.shape[2]-self.patch_size-1)

                xx = self.images[batch_idx, i : i+self.patch_size, j : j+self.patch_size, :]
                yy = self.masks[batch_idx, i : i+self.patch_size, j : j+self.patch_size, :]

                if self.augment:
                    augmentation = randrange(5)
                    if augmentation == 1: #rotate 90
                        xx, yy = np.rot90(xx), np.rot90(yy)
                    elif augmentation == 2: #flip
                        xx, yy = np.flipud(xx), np.flipud(yy)
                    elif augmentation == 3: #flip
                        xx, yy = np.fliplr(xx), np.fliplr(yy)
                    elif augmentation == 4: #scale
                        etha = np.random.uniform(1 / etha_max, etha_max) 
                        size = int(np.ceil(self.patch_size*etha))
                        i = np.random.choice(a=self.images.shape[1]-size-1)
                        j = np.random.choice(a=self.images.shape[2]-size-1)
                        xx = resize(self.images[batch_idx, i : i+size, j : j+size, :], (self.patch_size, self.patch_size))
                        yy = resize(self.masks[batch_idx, i : i+size, j : j+size, :], (self.patch_size, self.patch_size))

                batch_x.append(xx)
                batch_y.append(yy)

            batch_x = np.stack(batch_x)
            batch_y = np.stack(batch_y)

            yield (batch_x, batch_y)