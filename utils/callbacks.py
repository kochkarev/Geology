from pathlib import Path
from .eval import TestEvaluator, Tester

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


class TestResults(Callback):
    def __init__(self, images, masks, predict_func, output_path: Path, codes_to_lbls, lbls_to_colors, offset):
        self.images = images
        self.masks = masks
        self.predict_func = predict_func
        self.lrs = []
        self.evaluator = TestEvaluator(codes_to_lbls, offset)
        self.tester = Tester(self.evaluator, output_path, codes_to_lbls, lbls_to_colors)

    def on_epoch_end(self, epoch, logs=None):
        description = f'epoch {epoch + 1}'
        self.tester.test_on_set(list(zip(self.images, self.masks)), self.predict_func, description)
        self.lrs.append(K.eval(self.model.optimizer.lr))
        self.tester.plot_LR(self.lrs)
