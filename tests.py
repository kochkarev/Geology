from generators import PatchGenerator
from data_utils import get_imgs_masks
import os
import json
from tensorflow.keras.utils import to_categorical
from utils import colorize_mask, to_heat_map, create_heatmaps
from PIL import Image
import numpy as np
from skimage.transform.integral import integral_image
from scipy.ndimage.morphology import distance_transform_edt
from config import classes_mask

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

stat = dict()

def _test_balancing(output_path: str, patches_num: int, input_img: str, patch_size: str):

    classes = [cl for cl in classes_mask.values()]
    num_classes = len(classes)

    mask = np.array(Image.open(os.path.join("input", "dataset", input_img)))[:,:,0]

    input_img = input_img.replace('_NEW.png', '')
    stat[input_img] = {i : str(0) for i in classes}

    p_s = [(np.load(cl + "__" + input_img + ".npz")).f.arr_0 for cl in classes]

    for p, cl in zip(p_s, classes):

        aa = np.arange(mask.shape[0] * mask.shape[1])
        a=np.indices(mask.shape)

        for i in range(patches_num):
            n = np.random.choice(a=aa, p=np.ndarray.flatten(p))
            xx=np.ndarray.flatten(a[0])
            yy=np.ndarray.flatten(a[1])

            patch = mask[xx[n]:xx[n] + patch_size, yy[n]:yy[n] + patch_size]

            class_n = classes.index(cl)
            unique, counts = np.unique(mask, return_counts=True)
            aaa = dict(zip(unique, counts))
            if class_n in aaa:
                stat[input_img][cl] = str(aaa[class_n])
            patch = colorize_mask(np.dstack((patch, patch, patch)), n_classes=num_classes)

            Image.fromarray(patch.astype(np.uint8)).save(os.path.join(output_path, f"{cl}_{i + 1}_{input_img}.jpg"))

def test_balancing(output_path: str, patches_num: int, patch_size: int, generate: bool = False, statistics: bool = False):

    with open(os.path.join("input", "dataset.json")) as dataset_json:
        names = json.load(dataset_json)

    marked_images = names["BoxA_DS1"]["marked"]

    if generate:
        print('Generating heatmaps..')
        for mask in marked_images:
            create_heatmaps(4, patch_size, mask.replace(".jpg", "_NEW.png"), output_path, True)
        if statistics:
            with open(os.path.join(output_path, "stat_balanced_512.json"), 'w') as fp:
                json.dump(stat, fp, indent=4)

    for mask in marked_images:
        print(f'Making {patches_num} patches from {mask}')
        _test_balancing(output_path, patches_num, mask.replace(".jpg", "_NEW.png"), patch_size)

if __name__ == "__main__":
    # output_path = os.path.join("test_output")
    # os.makedirs(output_path, exist_ok=True)
    # test_augmentation(output_path, 200)

    output_path = os.path.join("test_output", "balancing")
    os.makedirs(output_path, exist_ok=True)
    test_balancing(output_path, 10, 512, generate=True, statistics=True)

    # create_heatmaps(4, 512, "Py-Cpy-Sh-BR-GL2_NEW.png", "", True)