from generators import PatchGenerator
from data_utils import get_imgs_masks
import os
from tensorflow.keras.utils import to_categorical
from utils import colorize_mask
from PIL import Image
import numpy as np

def test_augmentation(output_path: str, patches_num: int):

    path = os.path.join(os.path.dirname(__file__), "input", "dataset")

    x_train, _, y_train, _ = get_imgs_masks(path)
    y_train = to_categorical(y_train, num_classes=4)

    train_generator = PatchGenerator(images=x_train, masks=y_train, patch_size=512, batch_size=1, augment=True)
    aug_iter = iter(train_generator)

    for i in range(patches_num):

        img, mask = next(aug_iter)

        Image.fromarray((img[0] * 255).astype(np.uint8)).save(os.path.join(output_path, f'img_{i + 1}.jpg'))
        mask = np.argmax(mask[0], axis=2)
        Image.fromarray(
            colorize_mask(np.dstack((mask,mask,mask)), n_classes=4).astype(np.uint8)
        ).save(os.path.join(output_path, f'mask_{i + 1}.jpg')) 

if __name__ == "__main__":
    output_path = os.path.join("test_output")
    os.makedirs(output_path, exist_ok=True)

    test_augmentation(output_path, 200)