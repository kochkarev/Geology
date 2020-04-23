from generators import PatchGenerator
from data_utils import get_imgs_masks
import os
import json
from tensorflow.keras.utils import to_categorical
from utils import colorize_mask, to_heat_map, create_heatmap
from PIL import Image
import numpy as np
from skimage.transform.integral import integral_image
from scipy.ndimage.morphology import distance_transform_edt
from config import classes_mask
from time import time

stat = dict()

def test_augmentation(output_path: str, patches_num: int):

    classes = [cl for cl in classes_mask.values()]
    path = os.path.join(os.path.dirname(__file__), "input", "dataset")

    x_train, _, y_train, _, train_names, _ = get_imgs_masks(path, False, True)
    y_train = to_categorical(y_train, num_classes=4).astype(np.uint8)

    print('Initialization generator..')
    t1=time()
    train_generator = PatchGenerator(images=x_train, masks=y_train, names=train_names, patch_size=512, batch_size=8, augment=True)
    t2=time()
    print(f'took {t2-t1} seconds')
    aug_iter = iter(train_generator)

    for i in range(patches_num):

        print(f'\nGenerating patch {i+1}')
        t1= time()
        img, masks = next(aug_iter)
        t2=time()
        print(f'generating took {t2-t1} seconds')
        stat[str(i)] = {i : str(0) for i in classes}   

        for j in range(8):
            Image.fromarray((img[j] * 255).astype(np.uint8)).save(os.path.join(output_path, f'img_{i + 1}_{j+1}.jpg'))
            mask = np.argmax(masks[j], axis=2)
            Image.fromarray(
                colorize_mask(np.dstack((mask,mask,mask)), n_classes=4).astype(np.uint8)
            ).save(os.path.join(output_path, f'mask_{i + 1}_{j + 1}.jpg'))

    s = sum(train_generator.stat[key] for key in train_generator.stat.keys())
    for key in train_generator.stat.keys():
        print(train_generator.stat[key] / s)

def test_rotation(output_path: str):

    classes = [cl for cl in classes_mask.values()]
    path = os.path.join(os.path.dirname(__file__), "input", "dataset")

    x_train, _, y_train, _, train_names, _ = get_imgs_masks(path, False, True)
    y_train = to_categorical(y_train, num_classes=4).astype(np.uint8)

    print('Initialization generator..')
    t1=time()
    train_generator = PatchGenerator(images=x_train, masks=y_train, names=train_names, patch_size=512, batch_size=8, augment=True)
    t2=time()
    print(f'took {t2-t1} seconds')
    aug_iter = iter(train_generator)



# def _test_balancing(output_path: str, patches_num: int, input_img: str, patch_size: str):

#     classes = [cl for cl in classes_mask.values()]
#     num_classes = len(classes)

#     mask = np.array(Image.open(os.path.join("input", "dataset", input_img)))[:,:,0]

#     input_img = input_img.replace('_NEW.png', '')
#     stat[input_img] = {i : str(0) for i in classes}

#     p_s = [(np.load((os.path.join("input", "dataset", cl + "__" + input_img + ".npz"))))["arr_0"] for cl in classes]

#     for p, cl in zip(p_s, classes):

#         aa = np.arange(mask.shape[0] * mask.shape[1])
#         a=np.indices(mask.shape)

#         for i in range(patches_num):
#             n = np.random.choice(a=aa, p=np.ndarray.flatten(p))
#             xx=np.ndarray.flatten(a[0])
#             yy=np.ndarray.flatten(a[1])

#             patch = mask[xx[n]:xx[n] + patch_size, yy[n]:yy[n] + patch_size]

#             class_n = classes.index(cl)
#             unique, counts = np.unique(patch, return_counts=True)
#             aaa = dict(zip(unique, counts))
#             if class_n in aaa:
#                 stat[input_img][cl] = str(aaa[class_n])
#             patch = colorize_mask(np.dstack((patch, patch, patch)), n_classes=num_classes)

#             Image.fromarray(patch.astype(np.uint8)).save(os.path.join(output_path, f"{cl}_{i + 1}_{input_img}.jpg"))

# def test_balancing(output_path: str, patches_num: int, patch_size: int, generate: bool = False, statistics: bool = False):

#     with open(os.path.join("input", "dataset.json")) as dataset_json:
#         names = json.load(dataset_json)

#     marked_images = names["BoxA_DS1"]["marked"]

#     if generate:
#         print('Generating heatmaps..')
#         for mask in marked_images:
#             print(f'Creating heatmap for {mask}')
#             create_heatmap(4, patch_size, mask.replace(".jpg", "_NEW.png"), output_path, True)
    
#     # for mask in marked_images:
#     #     print(f'Making {patches_num} patches from {mask}')
#     #     _test_balancing(output_path, patches_num, mask.replace(".jpg", "_NEW.png"), patch_size)

#     if statistics:
#         with open(os.path.join(output_path, "stat_balanced_" + str(patch_size) + ".json"), 'w') as fp:
#             json.dump(stat, fp, indent=4)

if __name__ == "__main__": 
    output_path = os.path.join("test_output", "augment")
    os.makedirs(output_path, exist_ok=True)
    test_augmentation(output_path, 100)

    # output_path = os.path.join("test_output", "rotate")
    # os.makedirs(output_path, exist_ok=True)
    # test_rotation(output_path)

    # generate_lists_mineral()

    # output_path = os.path.join("test_output", "balancing_256_thr90")
    # os.makedirs(output_path, exist_ok=True)
    # test_balancing(output_path, 5, 256, generate=True, statistics=False)

    # create_heatmaps(4, 512, "Py-Cpy-Sh-BR-GL2_NEW.png", "", True)