from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from callbacks import TestResults
from metrics import iou_tf
from unet import res_unet, weightedLoss
from utils.base import get_loss_weights_2, get_squeeze_mappings, squeeze_mask, prepare_experiment
from utils.generators import AutoBalancedPatchGenerator, SimpleBatchGenerator
from utils.patches import combine_from_patches, split_into_patches


class Model:

    def __init__(self, patch_size, batch_size, offset, n_classes, LR, patch_overlay, class_weights=None):
        self.patch_s = patch_size
        self.batch_s = batch_size
        self.offset = offset
        self.n_classes = n_classes
        self.model = None
        self.LR = LR
        self.patch_overlay = patch_overlay
        self.class_weights = class_weights
        if self.class_weights is None:
            self.class_weights = [1 / self.n_classes] * self.n_classes

    def initialize(self, n_filters, n_layers):
        self.model = res_unet(
            (None, None, 3),
            n_classes=self.n_classes,
            BN=True,
            filters=n_filters,
            n_layers=n_layers,
        )

        # self.model.compile(
        #     optimizer=Adam(learning_rate=self.LR), 
        #     loss = weightedLoss(categorical_crossentropy, self.class_weights),
        #     metrics=[iou_tf]
        # )

    def load(self, model_path):
        # self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.load_weights(model_path)


    def predict_image(self, img: np.ndarray):
        patches = split_into_patches(img, self.patch_s, self.offset, self.patch_overlay)
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
        result = combine_from_patches(pred_patches, self.patch_s, self.offset, self.patch_overlay, img.shape[:2])
        return result

    def train(self, train_generator, val_generator, steps_per_epoch, epochs, validation_steps, test_data, test_output: Path, test_vis: bool, test_patch_overlay):
        
        callback_test = TestResults(
            images=test_data[0], 
            masks=test_data[1],
            names=test_data[2], 
            predict_func=lambda img: self.predict_image(img), 
            n_classes=self.n_classes,
            batch_size=self.batch_s,
            patch_size=self.patch_s,
            offset=self.offset,
            output_path=test_output,
            patch_overlay=test_patch_overlay,
            all_metrics=['iou', 'iou_strict', 'acc'],
            vis=test_vis
        )

        (test_output / 'models').mkdir()
        checkpoint_path = str(test_output / 'models' / 'best.hdf5')

        callback_checkpoint = ModelCheckpoint(monitor='val_loss', filepath=checkpoint_path,
                                             save_best_only=True, save_weights_only=True)
        
        _ = self.model.fit(
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


def load_test_data(imgs_path, masks_path, n_classes, squeeze=True, missed_classes=None):
    print('Loading test data')
    squeeze_mappings = get_squeeze_mappings(n_classes, missed_classes)
    n_classes_sq = n_classes - len(missed_classes)
    imgs, masks, names = [], [], []
    for p in sorted(list(imgs_path.iterdir())[:1]):
        img = np.array(Image.open(p)).astype(np.float32) / 256
        mask_p = masks_path / (p.stem + '.png')
        mask = np.array(Image.open(mask_p))
        if squeeze:
            mask = squeeze_mask(mask, squeeze_mappings)
        mask = to_categorical(mask[:, :, 0], n_classes_sq)
        imgs.append(img)
        masks.append(mask)
        names.append(p.name)
    print('Loading test data finished')
    return imgs, masks, names


n_classes = 13
n_classes_sq = 7
patch_s = 256
batch_s = 4 #12
n_layers = 4
n_filters = 8
LR = 0.001
patch_overlay = 0.5

missed_classes = (3, 5, 7, 9, 10, 12)

exp_path = prepare_experiment(Path('output'))


pg = AutoBalancedPatchGenerator(
    Path('c:\\dev\\#data\\LumenStone\\S1\\v1\\imgs\\train\\'),
    Path('c:\\dev\\#data\\LumenStone\\S1\\v1\\masks\\train\\'),
    Path('.\\cache\\maps\\'),
    patch_s, n_classes=n_classes, distancing=0.5, mixin_random_every=5)


# loss_weights = recalc_loss_weights_2(pg.get_class_weights(remove_missed_classes=True))

bg = SimpleBatchGenerator(pg, batch_s, n_classes_sq, squeeze_mask=True, augment=True)


test_data = load_test_data(
    Path('c:\\dev\\#data\\LumenStone\\S1\\v1\\imgs\\test\\'),
    Path('c:\\dev\\#data\\LumenStone\\S1\\v1\\masks\\test\\'),
    n_classes, squeeze=True, missed_classes=missed_classes
)

model = Model(patch_s, batch_s, offset=8, n_classes=n_classes_sq, LR=LR, patch_overlay=patch_overlay, class_weights=None)
model.initialize(n_filters, n_layers)
# model.load(Path('./output/exp_89/models/best.hdf5'))

model.model.compile(
            optimizer=Adam(learning_rate=LR), 
            # loss = weightedLoss(categorical_crossentropy, loss_weights),
            loss = categorical_crossentropy,
            metrics=[iou_tf]
        )

model.train(bg.g_balanced(), bg.g_random(), steps_per_epoch=1000, epochs=50, validation_steps=50, test_data=test_data,
            test_output=exp_path, test_vis=True, test_patch_overlay=patch_overlay)
