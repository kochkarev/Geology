from utils.data_generators.balancing import AutoBalancedPatchGenerator
from utils.data_generators.core import SimpleBatchGenerator, _squeeze_mask, recalc_loss_weights
import numpy as np
import tensorflow as tf
from utils.patches import combine_patches, split_to_patches
from PIL import Image
from pathlib import Path
from unet import custom_unet, weightedLoss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from metrics import iou_tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from callbacks import TestResults


class Model:

    def __init__(self, patch_size, batch_size, offset, n_classes, LR=0.001, class_weights=None):
        self.patch_s = patch_size
        self.batch_s = batch_size
        self.offset = offset
        self.n_classes = n_classes
        self.model = None
        self.LR = LR
        self.class_weights = class_weights
        if self.class_weights is None:
            self.class_weights = [1 / self.n_classes] * self.n_classes

    def initialize(self, n_filters, n_layers):
        self.model = custom_unet(
            (None, None, 3),
            n_classes=self.n_classes,
            filters=n_filters,
            use_batch_norm=True,
            n_layers=n_layers,
            output_activation='softmax'
        )

        self.model.compile(
            optimizer=Adam(learning_rate=self.LR), 
            loss = weightedLoss(categorical_crossentropy, self.class_weights),
            metrics=[iou_tf]
        )

    def load(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def predict_image(self, img: np.ndarray, overlay):
        patches = split_to_patches(img, self.patch_s, self.offset, overlay=overlay)
        init_patch_len = len(patches)

        while (len(patches) % self.batch_s != 0):
            patches.append(patches[-1])
        pred_patches = []

        for i in range(0, len(patches), self.batch_s):
            batch = np.stack(patches[i : i+self.batch_s])
            prediction = self.model.predict_on_batch(batch)
            for x in prediction:
                pred_patches.append(x)
        
        pred_patches = pred_patches[:init_patch_len]
        result = combine_patches(pred_patches, self.patch_s, self.offset, overlay=overlay,
                                 orig_shape=(img.shape[0], img.shape[1], pred_patches[0].shape[2]))
        return result

    def train(self, train_generator, val_generator, steps_per_epoch, epochs, validation_steps, test_data, test_overlay, test_output: Path, test_vis: bool):
        
        callback_test = TestResults(
            images=test_data[0], 
            masks=test_data[1],
            names=test_data[2], 
            predict_func=lambda img, overlay: self.predict_image(img, overlay), 
            n_classes=self.n_classes,
            batch_size=self.batch_s,
            patch_size=self.patch_s,
            overlay=test_overlay,
            offset=self.offset,
            output_path=test_output,
            all_metrics=['iou', 'iou_strict', 'acc'],
            vis=test_vis
        )

        # callbacks = [
        #     callback_test,
        #     EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4),
            # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=1e-4),
        # ]

        (test_output / 'models').mkdir()
        checkpoint_path = str(test_output / 'models' / 'best.hdf5')

        callback_checkpoint = ModelCheckpoint(monitor='val_loss', filepath=checkpoint_path,
                                             save_best_only=True, save_weights_only=True)
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator, 
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=[
                callback_test,
                callback_checkpoint,
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=1e-4),
            ],
        )


def load_test(imgs_path, masks_path, n_classes, squeeze=True):
    print('Loading test data')
    imgs, masks, names = [], [], []
    for p in list(imgs_path.iterdir()):
        img = np.array(Image.open(p)).astype(np.float32) / 256
        mask_p = masks_path / (p.stem + '.png')
        mask = np.array(Image.open(mask_p))
        if squeeze:
            mask = _squeeze_mask(mask)
        mask = to_categorical(mask[:, :, 0], n_classes)
        imgs.append(img)
        masks.append(mask)
        names.append(p.name)
    print('Loading test data finished')
    return imgs, masks, names


def prepare_experiment(out_path: Path) -> Path:
    out_path.mkdir(parents=True, exist_ok=True)
    dirs = list(out_path.iterdir())
    experiment_id = max(int(d.name.split('_')[1]) for d in dirs) + 1 if dirs else 1
    exp_path = out_path / f'exp_{experiment_id}'
    exp_path.mkdir()
    return exp_path


n_classes = 13
n_classes_sq = 7
patch_s = 256
batch_s = 6
n_layers = 4
n_filters = 8
LR = 0.001
 

pg = AutoBalancedPatchGenerator(
    Path('./data/LumenStone/S1/v1/imgs/train'),
    Path('./data/LumenStone/S1/v1/masks/train'),
    Path('./cache/maps'),
    patch_s, n_classes, distancing=0.0, prob_capacity=32)

loss_weights = recalc_loss_weights(pg.get_class_weights(remove_missed_classes=True))
print(f'Loss weights per class: {loss_weights}')

bg = SimpleBatchGenerator(pg, batch_s, n_classes_sq, squeeze_mask=True, augment=True)


test_data = load_test(
    Path('./data/LumenStone/S1/v1/imgs/test'),
    Path('./data/LumenStone/S1/v1/masks/test'),
    n_classes=n_classes_sq, squeeze=True
)

# model = Model(patch_s, batch_s, offset=8, n_classes=n_classes_sq, LR=LR, class_weights=None)
model = Model(patch_s, batch_s, offset=8, n_classes=n_classes_sq, LR=LR, class_weights=loss_weights)
model.initialize(n_filters, n_layers)


# # --- train model ---
exp_path = prepare_experiment(Path('output'))
# # model.train(bg.g(), bg.g(), steps_per_epoch=2000, epochs=100, validation_steps=200, test_data=test_data, test_overlay=0.0, test_output=exp_path)
model.train(bg.g(), bg.g(), steps_per_epoch=200, epochs=20, validation_steps=20, test_data=test_data, test_overlay=0.0, test_output=exp_path, test_vis=True)
