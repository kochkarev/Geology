import numpy as np

def PatchGenerator(images, masks, patch_size, batch_size):

    while(True):
    
        assert (type(images) is np.ndarray and type(masks) is np.ndarray), ("Input data type is expected to be ndarray")
        assert (images.shape == masks.shape), ("Original images and masks must be equal shape")

        batch_files = np.random.choice(a=images.shape[0], size=batch_size)

        batch_x = []
        batch_y = []

        for batch_idx in batch_files:

            i = np.random.choice(a=images.shape[1]-patch_size-1)
            j = np.random.choice(a=images.shape[2]-patch_size-1)

            xx = images[batch_idx, i : i+patch_size, j : j+patch_size, :]
            yy = masks[batch_idx, i : i+patch_size, j : j+patch_size, :]

            batch_x.append(xx)
            batch_y.append(yy)

        batch_x = np.stack(batch_x)
        batch_y = np.stack(batch_y)

        yield (batch_x, batch_y)