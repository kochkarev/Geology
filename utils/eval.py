from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from metrics import acc, iou, iou_per_class, to_strict
from tqdm import tqdm

from utils.vis import vis_segmentation


@dataclass
class EvaluationResult:
    iou_per_class: Dict[str, float]
    iou_per_class_strict: Dict[str, float]
    iou: float
    iou_strict: float
    acc: float
    pixels_per_class: Union[Dict[str, int], None]

    def to_str(self, description: str) -> str:
        iou_per_class_str = ''.join(f'\t\t {cl}: {iou:.4f}\n' for cl, iou in self.iou_per_class.items())
        iou_strict_per_class_str = ''.join(f'\t\t {cl}: {iou:.4f}\n' for cl, iou in self.iou_per_class_strict.items())
        s = (
            f'Evaluatoin result ({description}):\n'
            f'\t iou: {self.iou:.4f}\n'
            f'\t iou_strict: {self.iou_strict:.4f}\n'
            f'\t acc: {self.acc:.4f}\n'
            f'\t iou per class:\n'
            f'{iou_per_class_str}'
            f'\t iou_strict per class:\n'
            f'{iou_strict_per_class_str}'
        )
        return s


class TestEvaluator:

    def __init__(self, codes_to_lbls, offset) -> None:
        self.codes_to_lbls = codes_to_lbls
        self.offset = offset
        self.buffer: List[EvaluationResult] = []
        self.archive: List[EvaluationResult] = []

    def evaluate(self, pred: np.ndarray, gt: np.ndarray) -> EvaluationResult:
        if self.offset > 0:
            gt_cr = gt[self.offset : -self.offset, self.offset : -self.offset, ...]
            pred_cr = pred[self.offset : -self.offset, self.offset : -self.offset, ...]

        def to_dict(metric_vals):
            return {self.codes_to_lbls[i]: v for i, v in enumerate(metric_vals)}

        pixels_per_class = {self.codes_to_lbls[i]: np.sum(gt_cr[..., i]) for i in range(gt_cr.shape[-1])}
        pred_strict = to_strict(pred_cr)
        eval_res = EvaluationResult(
            iou_per_class=to_dict(iou_per_class(gt_cr, pred_cr)),
            iou_per_class_strict=to_dict(iou_per_class(gt_cr, pred_strict)),
            iou=iou(gt_cr, pred_cr),
            iou_strict=iou(gt_cr, pred_strict),
            acc=acc(gt_cr, pred_strict),
            pixels_per_class=pixels_per_class
        )
        self.buffer.append(eval_res)        
        return eval_res

    def flush(self) -> EvaluationResult:
        n = len(self.buffer)
        total_iou_pc = dict()
        total_iou_pc_strict = dict()

        for _, cl in self.codes_to_lbls.items():
            px_per_image = np.array([e.pixels_per_class[cl] for e in self.buffer])
            s = np.sum(px_per_image)
            px_per_image = px_per_image / s if s > 0 else px_per_image
            total_iou_pc[cl] = sum(self.buffer[i].iou_per_class[cl] * px_per_image[i] for i in range(n))
            total_iou_pc_strict[cl] = sum(self.buffer[i].iou_per_class_strict[cl] * px_per_image[i] for i in range(n))

        total_iou = sum(e.iou for e in self.buffer) / n
        total_iou_strict = sum(e.iou_strict for e in self.buffer) / n
        total_acc = sum(e.acc for e in self.buffer) / n
        
        current_eval_res = EvaluationResult(
            iou_per_class=total_iou_pc, iou_per_class_strict=total_iou_pc_strict,
            iou=total_iou, iou_strict=total_iou_strict,
            acc=total_acc,
            pixels_per_class=None
        )

        self.buffer.clear()
        self.archive.append(current_eval_res)
        return current_eval_res

    def _get_values(self, metric_name):
        metric_vals = [getattr(e, metric_name) for e in self.archive]
        classes_str = self.codes_to_lbls.values()
        return {cl_str: [m[cl_str] for m in metric_vals] for cl_str in classes_str}

    def get_plot_data(self) -> Tuple[List, List]:
        single_class_plot_data = [
            ('acc', [e.acc for e in self.archive]),
            ('iou', [e.iou for e in self.archive]),
            ('iou_strict', [e.iou_strict for e in self.archive]),
        ]
        multi_class_plot_data = [
            ('iou', self._get_values('iou_per_class')),
            ('iou_strict', self._get_values('iou_per_class_strict')),
        ]
        return single_class_plot_data, multi_class_plot_data


class Tester:
    
    def __init__(self, evaluator, out_path: Path, codes_to_lbls, lbls_to_colors):
        self.evaluator = evaluator
        self.do_visualization = True
        self.out_path = out_path
        self.codes_tp_lbls = codes_to_lbls
        self.lbls_to_colors = lbls_to_colors
        self.codes_to_colors = {code: lbls_to_colors[lbl] for code, lbl in codes_to_lbls.items()}

    def _visualize(self, img, gt, pred, folder: Path, image_idx):
        if self.do_visualization and img is not None:
            img = (img * 255).astype(np.uint8)
            mask = np.argmax(gt, axis=-1).astype(np.uint8)
            pred = np.argmax(pred, axis=-1).astype(np.uint8)
            img_name = f'img_{image_idx}'
            vis_segmentation(img, mask, pred, self.evaluator.offset, self.codes_to_colors, folder, img_name)

    def test_on_set(self, data: Iterable[Tuple[np.ndarray, np.ndarray]], predict_func, description: str) -> EvaluationResult:
        out_folder = (self.out_path / description)
        out_folder.mkdir(exist_ok=True, parents=True)
        log = open(self.out_path / 'metrics.txt', "a+")
        log_detailed = open(self.out_path / 'metrics_detailed.txt', "a+")
        for i, (img, mask) in enumerate(tqdm(data, 'testing')):
            pred = predict_func(img)
            eval_res = self.evaluator.evaluate(pred, mask)
            eval_res_str = eval_res.to_str(description=f'{description}, image {i + 1}')
            # print(eval_res_str)
            log_detailed.write(eval_res_str + '\n')
            self._visualize(img, mask, pred, out_folder, i + 1)
        total_eval_res = self.evaluator.flush()
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