import numpy as np

class PatchGenerator:

    def __init__(self, images : np.ndarray, masks : np.ndarray, patch_size : int, batch_size : int):

        self.images = images
        self.masks = masks
        self.patch_size = patch_size
        self.batch_size = batch_size

    def __iter__(self):

        while(True):
        
            assert (self.images.shape[:3] == self.masks.shape[:3]), ("Original images and masks must be equal shape")

            batch_files = np.random.choice(a=self.images.shape[0], size=self.batch_size)

            batch_x = []
            batch_y = []

            for batch_idx in batch_files:

                i = np.random.choice(a=self.images.shape[1]-self.patch_size-1)
                j = np.random.choice(a=self.images.shape[2]-self.patch_size-1)

                xx = self.images[batch_idx, i : i+self.patch_size, j : j+self.patch_size, :]
                yy = self.masks[batch_idx, i : i+self.patch_size, j : j+self.patch_size, :]

                batch_x.append(xx)
                batch_y.append(yy)

            batch_x = np.stack(batch_x)
            batch_y = np.stack(batch_y)

            yield (batch_x, batch_y)