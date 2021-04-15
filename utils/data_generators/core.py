from typing import List
import numpy as np
from tensorflow.keras.utils import to_categorical

mapping = {
    0: 0,
    1: 1,
    2: 2,
    4: 3,
    6: 4,
    8: 5,
    11: 6
}


def _squeeze_mask(mask: np.ndarray):
    new_mask = np.zeros_like(mask)
    for i, j in mapping.items():
        new_mask[mask == i] = j
    return new_mask


def recalc_loss_weights(class_weights: List) -> List:
    assert not any(w == 0 for w in class_weights)
    loss_w = [1 / w for w in class_weights]
    s = sum(loss_w)
    loss_w = [w / s for w in loss_w]
    return loss_w


def recalc_loss_weights_2(class_weights: List, delta=0.5) -> List:
    assert not any(w == 0 for w in class_weights)
    w_src = class_weights
    w_dst = [1 / w for w in class_weights]
    w_res = [w_src[i] + (w_dst[i] - w_src[i]) * delta for i in range(len(w_src))]
    s = sum(w_res)
    return [w / s for w in w_res]

class SimpleBatchGenerator:

    def __init__(self, patch_generator, batch_s, n_classes, squeeze_mask, augment=True) -> None:
        self.patch_generator = patch_generator
        self.batch_s = batch_s
        self.n_classes = n_classes
        self.squeeze_mask = squeeze_mask
        self.augment = augment

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

    def g(self, random):
        x, y = [], []
        while True:
            if not random:
                img, mask, _ = self.patch_generator.get_patch()
            else:
                img, mask, _ = self.patch_generator.get_patch_random(update_accumulators=False)
            if self.squeeze_mask:
                mask = _squeeze_mask(mask)
            mask = to_categorical(mask, self.n_classes)
            if self.augment:
                img, mask = self._augment(img, mask)
            x.append(img)
            y.append(mask)
            if len(x) == self.batch_s:
                yield(np.stack(x),  np.stack(y))
                x.clear()
                y.clear()

    def g_random(self):
        return self.g(True)

    def g_balanced(self):
        return self.g(False)
