from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.backend import squeeze


@dataclass
class MaskLoadParams:
    n_classes: int
    squeeze: True
    squeeze_mappings: Dict[int, int]

    def __post_init__(self):
        self.n_classes = self.n_classes if not squeeze else len(self.squeeze_mappings)


def squeeze_mask(mask: np.ndarray, mapping):
    new_mask = np.zeros_like(mask)
    for i, j in mapping.items():
        new_mask[mask == i] = j
    return new_mask


def prepocess_mask(mask: np.ndarray, params: MaskLoadParams):
    if params.squeeze:
        mask = squeeze_mask(mask, params.squeeze_mappings)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = to_categorical(mask, params.n_classes)
    return mask


def get_loss_weights(class_weights: List) -> List:
    assert not any(w == 0 for w in class_weights)
    loss_w = [1 / w for w in class_weights]
    s = sum(loss_w)
    loss_w = [w / s for w in loss_w]
    return loss_w


def get_loss_weights_2(class_weights: List, delta=0.5) -> List:
    assert not any(w == 0 for w in class_weights)
    w_src = class_weights
    w_dst = [1 / w for w in class_weights]
    w_res = [w_src[i] + (w_dst[i] - w_src[i]) * delta for i in range(len(w_src))]
    s = sum(w_res)
    return [w / s for w in w_res]


def prepare_experiment(out_path: Path) -> Path:
    out_path.mkdir(parents=True, exist_ok=True)
    dirs = list(out_path.iterdir())
    dirs = [d for d in dirs if d.name.startswith('exp_')]
    experiment_id = max(int(d.name.split('_')[1]) for d in dirs) + 1 if dirs else 1
    exp_path = out_path / f'exp_{experiment_id}'
    exp_path.mkdir()
    return exp_path


def set_gpu(gpu_index):
	import tensorflow as tf
	physical_devices  = tf.config.experimental.list_physical_devices('GPU')
	print(f'Available GPUs: {len(physical_devices )}')
	if physical_devices:
		print(f'Choosing GPU #{gpu_index}')
		try:
			tf.config.experimental.set_visible_devices([physical_devices[gpu_index]], 'GPU')
			logical_devices = tf.config.list_logical_devices('GPU')
			assert len(logical_devices) == 1
			print(f'Success. Now visible GPUs: {len(logical_devices)}')
		except RuntimeError as e:
			print('Something went wrong!')
			print(e)
