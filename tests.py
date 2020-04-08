from generators import PatchGenerator
from data_utils import get_imgs_masks
import os
from tensorflow.keras.utils import to_categorical
from utils import colorize_mask, to_heat_map
from PIL import Image
import numpy as np
from skimage.transform.integral import integral_image
from scipy.ndimage.morphology import distance_transform_edt

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

def test_balancing(output_path: str, patches_num: int):

    num_classes = 4
    patch_size = 512
    classes = ["Bg", "Sh", "PyMrc", "Gl"]
    mask = np.array(Image.open("input/dataset/Py-Cpy-Sh-BR-GL2_NEW.png"))[:,:,0]
    masks = to_categorical(mask, num_classes=num_classes, dtype=np.uint8)
    # dts = [distance_transform_edt(1-masks[:,:,i]) for i in range(num_classes)]
    # for dt, cl in zip(dts, classes):
    #     dt /= np.max(dt)
    #     dt = 1 - dt
    #     Image.fromarray(to_heat_map(dt)).save(os.path.join(output_path, f"dt_{cl}.jpg"))
    dts = []
    for i in range(num_classes):
        dt = distance_transform_edt(1-masks[:,:,i])
        dt /= np.max(dt)
        dt = 1 - dt
        dts.append(dt)
        Image.fromarray(to_heat_map(dt)).save(os.path.join(output_path, f"dt_{classes[i]}.jpg"))
        
    integrals = [np.pad(integral_image(dt), [(1,1),(1,1)], mode='constant') for dt in dts]

    # p_s = [np.zeros_like(integral) for integral in integrals]
    p_s = []
    
    for integral, cl in zip(integrals, classes):
        p = np.zeros_like(integral)
        for i in range(p.shape[0] - patch_size - 1):
            for j in range(p.shape[1] - patch_size - 1):
                p[i, j] = integral[i, j] + integral[i + patch_size,j + patch_size] - integral[i + patch_size, j] - integral[i, j + patch_size]
        
        p = p[:-patch_size - 1, :-patch_size - 1]
        max_p = np.max(p)
        min_p = np.min(p)
        p = p - min_p
        p = p / (max_p - min_p)
        p = p ** 2.5
        p = np.pad(p, [(0, patch_size - 1), (0, patch_size - 1)], mode='constant')
        Image.fromarray(to_heat_map(p)).save(os.path.join(output_path, f"heatmap_{cl}.jpg"))
        p = p / np.sum(p)
        p_s.append(p)

    for p, cl in zip(p_s, classes):

        aa = np.arange(mask.shape[0] * mask.shape[1])
        a=np.indices(mask.shape)

        for i in range(patches_num):
            n = np.random.choice(a=aa, p=np.ndarray.flatten(p))
            xx=np.ndarray.flatten(a[0])
            yy=np.ndarray.flatten(a[1])

            patch = mask[xx[n]:xx[n] + patch_size, yy[n]:yy[n] + patch_size]
            patch = colorize_mask(np.dstack((patch, patch, patch)), n_classes=num_classes)

            Image.fromarray(patch.astype(np.uint8)).save(os.path.join(output_path, f'{cl}_{i + 1}.jpg'))

if __name__ == "__main__":
    # output_path = os.path.join("test_output")
    # os.makedirs(output_path, exist_ok=True)

    # test_augmentation(output_path, 200)

    output_path = os.path.join("test_output", "balancing")
    os.makedirs(output_path, exist_ok=True)
    
    test_balancing(output_path, 10)