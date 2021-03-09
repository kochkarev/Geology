from pathlib import Path
from utils.generators import _squeeze_mask
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


class SimpleDataGenerator:

    def __init__(self, path: Path, batch_s, n_classes, squeeze=True, val_ratio=0.2, augment=True, preload=False):
        self.path = path
        self.batch_s = batch_s
        self.n_classes = n_classes
        self.squeeze = squeeze
        self.val_ratio = 0.2
        self.augment = augment
        self.preload = preload
        self.paths = list(self.path.iterdir())
        n_train = int((len(self.paths) // 2) * (1 - val_ratio))
        # n_val = len(self.paths) // 2 - n_train
        self.paths_train = self._extrat_pairs(self.paths[: 2 * n_train])
        self.paths_val = self._extrat_pairs(self.paths[2 * n_train : ])
        self.data_train = None
        self.data_val = None
        if self.preload:
            self.data_train = self.preload_data(self.paths_train)
            self.data_val = self.preload_data(self.paths_val)

    def _augment(self, x: np.ndarray, y: np.ndarray):
        n_rot = np.random.randint(0, 4)
        x = np.rot90(x, n_rot)
        y = np.rot90(y, n_rot)
        if np.random.randint(2) == 0:
            x = np.flip(x, 0)
            y = np.flip(y, 0)
        if np.random.randint(2) == 0:
            x = np.flip(x, 1)
            y = np.flip(y, 1)
        return x, y

    def _extrat_pairs(self, paths):
        print('###: ', (len(paths) // 2))
        res_paths = [None] * (len(paths) // 2)
        i = 0
        for p in paths:
            if p.stem.endswith('_m'):
                continue
            img_p = p
            mask_p = p.parent / p.name.replace('.png', '_m.png')
            print(i, img_p, mask_p)
            if i >= (len(paths) // 2):
                print(i)
            else:
                res_paths[i] = (img_p, mask_p)
            i += 1
        return res_paths

    def load_img_mask(self, img_p, mask_p):
        img = np.array(Image.open(img_p))
        mask = np.array(Image.open(mask_p))
        if self.squeeze:
            mask = _squeeze_mask(mask)
        mask = to_categorical(mask, self.n_classes)
        return img, mask

    def preload_data(self, paths):
        data = [None] * len(paths)
        for i, (img_p, mask_p) in tqdm(enumerate(paths)):
            data[i] = self.load_img_mask(img_p, mask_p)
        return data

    def generator(self, paths, data):
        x, y = [], []
        while True:
            for i, (img_p, mask_p) in enumerate(paths):
                if self.preload:
                    img, mask = data[i]
                else:
                    img, mask = self.load_img_mask(img_p, mask_p)
                if self.augment:
                    img, mask = self._augment(img, mask)
                x.append(img)
                y.append(mask)
                if len(x) == self.batch_s:
                    yield(np.stack(x), np.stack(y))
                    x.clear()
                    y.clear()

    def train_generator(self):
        return self.generator(self.paths_train, self.data_train)


    def val_generator(self):
        return self.generator(self.paths_val, self.data_val)

