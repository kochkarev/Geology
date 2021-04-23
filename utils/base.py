from pathlib import Path
from typing import Iterable, List

import numpy as np


def get_squeeze_mappings(n_classes: int, missed_classes: Iterable[int]):
    if missed_classes is None:
        missed_classes = []
    old_classes = tuple(i for i in range(n_classes) if i not in missed_classes)
    new_classes = tuple(range(n_classes - len(missed_classes)))
    mapping = {cl: new_classes[i]  for i, cl in enumerate(old_classes)}
    return mapping


def squeeze_mask(mask: np.ndarray, mapping):
    new_mask = np.zeros_like(mask)
    for i, j in mapping.items():
        new_mask[mask == i] = j
    return new_mask


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
