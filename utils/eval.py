import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from metrics import acc, iou_per_class, to_strict
from PIL import Image
from tqdm import tqdm

from .base import MaskLoadParams, prepocess_mask
from .vis import vis_segmentation

from metrics import exIoU, exAcc, joint_iou, joint_acc


@dataclass
class EvaluationResult:
    iou_class: Dict[str, exIoU]
    iou_class_strict: Dict[str, exIoU]
    mean_iou: float
    mean_iou_strict: float
    acc: exAcc

    def to_str(self, description: str) -> str:
        iou_cl_str = ''.join(f'\t\t {cl}: {iou.iou:.4f}\n' for cl, iou in self.iou_class.items())
        iou_cl_strict_str = ''.join(f'\t\t {cl}: {iou.iou:.4f}\n' for cl, iou in self.iou_class_strict.items())
        s = (
            f'Evaluatoin result ({description}):\n'
            f'\t mean iou: {self.mean_iou:.4f}\n'
            f'\t mean iou (strict): {self.mean_iou_strict:.4f}\n'
            f'\t acc: {self.acc.acc:.4f}\n'
            f'\t iou per class:\n'
            f'{iou_cl_str}'
            f'\t iou (strict) per class:\n'
            f'{iou_cl_strict_str}'
        )
        return s


class TestEvaluator:

    def __init__(self, codes_to_lbls, offset) -> None:
        self.codes_to_lbls = codes_to_lbls
        self.offset = offset
        self.buffer: List[EvaluationResult] = []
        self.archive: List[EvaluationResult] = []

    @staticmethod
    def _mean_iou(ious: Iterable[exIoU], weights=None) -> float:
        assert weights is None, 'not implemented yet'
        mean = sum(iou.iou for iou in ious) / len(ious)
        return mean

    def evaluate(self, pred: np.ndarray, gt: np.ndarray) -> EvaluationResult:
        if self.offset > 0:
            gt_cr = gt[self.offset : -self.offset, self.offset : -self.offset, ...]
            pred_cr = pred[self.offset : -self.offset, self.offset : -self.offset, ...]
        pred_strict = to_strict(pred_cr)

        def to_dict(metric_vals) -> Dict[str, exIoU]:
            return {self.codes_to_lbls[i]: v for i, v in enumerate(metric_vals)}

        iou_class = to_dict(iou_per_class(gt_cr, pred_cr))
        iou_class_strict = to_dict(iou_per_class(gt_cr, pred_strict))
        mean_iou = self._mean_iou(iou_class.values())
        mean_iou_strict = self._mean_iou(iou_class_strict.values())                 

        eval_res = EvaluationResult(
            iou_class=iou_class,
            iou_class_strict=iou_class_strict,
            mean_iou=mean_iou,
            mean_iou_strict=mean_iou_strict,
            acc=acc(gt_cr, pred_strict)
        )
        self.buffer.append(eval_res)        
        return eval_res

    def flush(self) -> EvaluationResult:
        total_iou_pc = dict()
        total_iou_pc_strict = dict()

        for cl in self.codes_to_lbls.values():
            total_iou_pc[cl] = joint_iou([e.iou_class[cl] for e in self.buffer])
            total_iou_pc_strict[cl] = joint_iou([e.iou_class_strict[cl] for e in self.buffer])

        total_iou = self._mean_iou(total_iou_pc.values())
        total_iou_strict = self._mean_iou(total_iou_pc_strict.values())
        total_acc = joint_acc([e.acc for e in self.buffer])

        current_eval_res = EvaluationResult(
            iou_class=total_iou_pc,
            iou_class_strict=total_iou_pc_strict,
            mean_iou=total_iou,
            mean_iou_strict=total_iou_strict,
            acc=total_acc
        )

        self.buffer.clear()
        self.archive.append(current_eval_res)
        return current_eval_res

    def _get_values(self, metric_name, sub_name):
        metric_vals = [getattr(e, metric_name) for e in self.archive]
        classes_str = self.codes_to_lbls.values()
        return {cl_str: [getattr(m[cl_str], sub_name) for m in metric_vals] for cl_str in classes_str}

    def get_plot_data(self) -> Tuple[List, List]:
        single_class_plot_data = [
            ('acc', [e.acc.acc for e in self.archive]),
            ('mean_iou', [e.mean_iou for e in self.archive]),
            ('mean_iou_strict', [e.mean_iou_strict for e in self.archive]),
        ]
        multi_class_plot_data = [
            ('iou', self._get_values('iou_class', 'iou')),
            ('iou_strict', self._get_values('iou_class_strict', 'iou')),
        ]
        return single_class_plot_data, multi_class_plot_data


class Tester:
    
    def __init__(self, evaluator, out_path: Path, codes_to_lbls, lbls_to_colors, mask_load_p: MaskLoadParams):
        self.evaluator = evaluator
        self.do_visualization = True
        self.out_path = out_path
        self.lbls_to_colors = lbls_to_colors
        self.codes_to_colors = {code: lbls_to_colors[lbl] for code, lbl in codes_to_lbls.items()}
        self.mask_load_p = mask_load_p

    def _load_test_pair(self, img_path: Path, mask_path: Path):
        img = np.array(Image.open(img_path)).astype(np.float32) / 256
        mask = np.array(Image.open(mask_path))
        mask = prepocess_mask(mask, self.mask_load_p)
        return img, mask

    def _visualize(self, img, gt, pred, folder: Path, image_idx):
        if self.do_visualization and img is not None:
            img = (img * 255).astype(np.uint8)
            mask = np.argmax(gt, axis=-1).astype(np.uint8)
            pred = np.argmax(pred, axis=-1).astype(np.uint8)
            img_name = f'img_{image_idx}'
            vis_segmentation(img, mask, pred, self.evaluator.offset, self.codes_to_colors, folder, img_name)

    def test_on_set(self, imgs_folder: Path, masks_folder: Path, predict_func, description: str) -> EvaluationResult:
        out_folder = (self.out_path / description)
        out_folder.mkdir(exist_ok=True, parents=True)
        log = open(self.out_path / 'metrics.txt', "a+")
        log_detailed = open(self.out_path / 'metrics_detailed.txt', "a+")
        img_paths = sorted(list(imgs_folder.iterdir()))
        mask_paths = [masks_folder / (img_path.stem + '.png') for img_path in img_paths]
        n = len(img_paths)
        for i in tqdm(range(n), 'testing'):
            img, mask = self._load_test_pair(img_paths[i], mask_paths[i])
            pred = predict_func(img)
            eval_res = self.evaluator.evaluate(pred, mask)
            eval_res_str = eval_res.to_str(description=f'{description}, image {i + 1}')
            log_detailed.write(eval_res_str + '\n')
            self._visualize(img, mask, pred, out_folder, i + 1)
        total_eval_res = self.evaluator.flush()
        gc.collect()
        self._redraw_metric_plots()
        total_eval_res_str = total_eval_res.to_str(description=f'{description}, total')
        print(total_eval_res_str)
        log_detailed.write(total_eval_res_str + '\n')
        log.write(total_eval_res_str + '\n')

    def _plot_single_class_metric(self, metric_name, values):
        epochs = len(values)
        fig = plt.figure(figsize=(12,6))
        # ax = plt.axes()
        # ax.set_facecolor('white')
        x = [x+1 for x in range(epochs)]
        y = [values[i] for i in range(epochs)]
        plt.plot(x, y)
        # plt.suptitle(f'{metric_name} over epochs', fontsize=20)
        plt.ylabel(f'{metric_name}', fontsize=20)
        plt.xlabel('epoch', fontsize=20)
        fig.savefig(self.out_path / f'{metric_name}.png')

    def _plot_multi_class_metric(self, metric_name, data: Dict[str, Iterable[float]]):
        epochs = len(list(data.values())[0])
        fig = plt.figure(figsize=(12,6))
        # ax = plt.axes()
        # ax.set_facecolor('white')
        for cl, vals in data.items():
            x = [x+1 for x in range(epochs)]
            y = [vals[i] for i in range(epochs)]
            plt.plot(x, y, color=self.lbls_to_colors[cl])
        # plt.suptitle(f'{metric_name} per class over epochs', fontsize=20)
        plt.ylabel(f'{metric_name}', fontsize=20)
        plt.xlabel('epoch', fontsize=20)
        plt.legend([cl_str for cl_str in data], loc='center right', fontsize=15)
        fig.savefig(self.out_path / f'{metric_name}_per_class.png')

    def _redraw_metric_plots(self):
        single_class_plot_data, multi_class_plot_data = self.evaluator.get_plot_data()
        for metric, data in single_class_plot_data:
            self._plot_single_class_metric(metric, data)
        for metric, data in multi_class_plot_data:
            self._plot_multi_class_metric(metric, data)

    def plot_LR(self, lrs):
        self._plot_single_class_metric('LR', lrs)
