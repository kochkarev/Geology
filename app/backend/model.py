import numpy as np
import tensorflow as tf
from utils import combine_patches, split_to_patches
from PIL import Image


class Model:

    def __init__(self, model_path, patch_size, batch_size, offset):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.offset = offset

    def predict(self, img):
        patches = split_to_patches(img, self.patch_size, self.offset, overlay=0.25)
        init_patch_len = len(patches)

        while (len(patches) % self.batch_size != 0):
            patches.append(patches[-1])
        pred_patches = []

        for i in range(0, len(patches), self.batch_size):

            batch = np.stack(patches[i : i+self.batch_size])
            prediction = self.model.predict_on_batch(batch)

            for x in prediction:
                pred_patches.append(x)
        
        pred_patches = pred_patches[:init_patch_len]
        result = combine_patches(pred_patches, self.patch_size, self.offset, overlay=0.25, orig_shape=(img.shape[0], img.shape[1], pred_patches[0].shape[2]))

        return result



img = np.array(Image.open('..\\sample_data\\Cpy-Sh-GL32.jpg')).astype(np.float32) / 255
print(img.shape)


model = Model('.\\models\\model_46_0.07.hdf5', 256, 32, 2 * 4)
p = model.predict(img)
p = np.argmax(p, axis=2)
print(p.shape)
Image.fromarray(p.astype(np.uint8) * 50).show()