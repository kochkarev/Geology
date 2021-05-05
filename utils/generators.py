import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import skimage.measure
from PIL import Image
from skimage.transform.integral import integral_image
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from .base import squeeze_mask
from .vis import to_heat_map


class AutoBalancedPatchGenerator:

    def __init__(self, img_dir_path: Path, mask_dir_path: Path, cache_path: Path, patch_size: int, n_classes: int,
                 prob_downscale_factor: int = 1, prob_downscale_func='max',
                 distancing=0.5, choose_strict_minority_class=False,
                 alpha=2, beta=3, hash_length=8, mixin_random_every=0, vis_path=None, quiet=True):
        assert prob_downscale_factor in (1, 2, 3, 4), f'downscale factor can be 1, 2, 3, 4'
        assert prob_downscale_func in ('max', 'mean')
        assert alpha > 1, 'alpha value should be > 1'
        self.image_dir_path = img_dir_path
        self.mask_dir_path = mask_dir_path
        self.cache_path = cache_path
        cache_path.mkdir(parents=True, exist_ok=True)
        self.patch_s = patch_size
        self.n_classes = n_classes
        self.prob_dx_factor = prob_downscale_factor
        self.prob_dx_func = prob_downscale_func
        self.distancing = distancing
        self.choose_strict_minority_class = choose_strict_minority_class
        self.alpha = alpha
        self.beta = beta
        self.hash_length = hash_length
        self.mixin_random_every = mixin_random_every
        self.vis_path = vis_path
        # --- perform initialization ---
        print('Initializing patch generator...')
        self.img_paths = sorted(list(img_dir_path.iterdir()))
        self.mask_paths = sorted(list(mask_dir_path.iterdir()))
        assert len(self.img_paths) == len(self.mask_paths), 'number of masks is not equal to number of imgs'
        # --- load all images, masks and prob maps ---
        self.imgs = self._load_imgs()
        self.masks = self._load_masks()
        self.prob_maps = self._load_maps(quiet=quiet)
        self.n_imgs = len(self.mask_paths)
        # --- calculate weights of images ---
        self.image_weights = self._calc_image_weights(self.masks)
        self.missed_classes = self._search_for_missed_classes(self.image_weights)
        print(f'\t missed classes: {self.missed_classes}')
        # --- initialize control of accumulated data ---
        self.patch_count = 0
        self.accumulated_patches = np.zeros([self.n_classes, self.n_imgs], dtype=np.uint8)
        self.accumulated_patches_per_image = [0] * self.n_imgs
        self.accumulated_px_per_class = [0] * n_classes
        self.accumulated_image_maps = [np.zeros(img.shape[:2], dtype=np.uint32) for img in self.imgs]
        # --- initialize benchmarks timing ---
        self._t_class_select = 0
        self._t_img_select = 0
        self._t_patch_extract = 0
        self._t_update_accumulators = 0
        print('Initializing patch generator finished.')

    def _load_imgs(self):
        print('\t Loading images...')
        return [np.array(Image.open(p)).astype(np.float32) / 255 for p in tqdm(self.img_paths)]
    
    def _load_masks(self):
        print('\t Loading masks...')
        return [self._load_mask(p) for p in tqdm(self.mask_paths)]
    
    def _load_maps(self, quiet):
        print('\t Loading probability maps...')
        maps = [self._get_prob_map(p, quiet) for p in tqdm(self.mask_paths)]
        gc.collect()
        print('\t Postprocessing probability maps...')
        maps = [self._postprocess_prob_map(pm, distancing=self.distancing) for pm in tqdm(maps)]
        return maps

    def _calc_image_weights(self, masks) -> np.ndarray:
        print('\t Calculating weights of images...')
        image_weights = np.zeros([self.n_classes, self.n_imgs], dtype=np.float)
        for cl in tqdm(range(self.n_classes)):
            image_weights[cl] = [np.sum(np.where(mask == cl, 1, 0)) for mask in masks]
            s = np.sum(image_weights[cl])
            if s > 0:
                image_weights[cl] = (image_weights[cl] / s) ** self.beta
                image_weights[cl] /= np.sum(image_weights[cl])
        return image_weights

    def _search_for_missed_classes(self, image_weights: np.ndarray):
        missed = []
        for cl in range(self.n_classes):
            if np.sum(image_weights[cl]) == 0:
                missed.append(cl)
        return missed

    def get_missed_classes(self):
        return self.missed_classes

    @staticmethod
    def _downscale_prob_map(prob_map: np.ndarray, factor: int, pool_function='mean'):
        assert pool_function in ('mean', 'max')
        if factor == 1:
            return prob_map
        f = {'mean': np.mean, 'max': np.max}[pool_function]
        n = len(prob_map)
        pm_downscaled = [None] * n
        for i in range(n):
            if prob_map[i] is not None:
                pm_downscaled[i] = skimage.measure.block_reduce(prob_map[i], (factor, factor), f)
                pm_downscaled[i] /= np.sum(pm_downscaled[i])
        return pm_downscaled

    def _get_prob_map(self, mask_path: Path, quiet=True):
        self.cache_path.mkdir(exist_ok=True)
        hash = int(hashlib.sha256(str(mask_path).encode('utf-8')).hexdigest(), 16) % (10 ** self.hash_length)
        downscale_suffix = '' if self.prob_dx_factor == 1 else f'_dx{self.prob_dx_factor}_{self.prob_dx_func}'
        prob_name_full = f'{hash}_{self.patch_s}.npz'
        prob_name_downscaled = f'{hash}_{self.patch_s}{downscale_suffix}.npz'
        if (self.cache_path / prob_name_downscaled).exists():
            # --- prob map at needed scale found in cache ---
            return np.load(self.cache_path / prob_name_downscaled, allow_pickle=True)['prob_maps']
        elif (self.cache_path / prob_name_full).exists():
            # --- prob map at needed scale not found in cache, but full prob map exists ---
            if not quiet:
                print(f'\t\t Probability map for {mask_path} found in cache, but at a wrong scale. Calculating...')
            pm_full = np.load(self.cache_path / prob_name_full, allow_pickle=True)['prob_maps']
            pm_downscaled = self._downscale_prob_map(pm_full, self.prob_dx_factor)
            np.savez_compressed(self.cache_path / prob_name_downscaled, prob_maps=pm_downscaled)
            return pm_downscaled
        else:
            if not quiet:
                print(f'\t\t Probability map for {mask_path} not found in cache. Calculating...')
            pm_full = self.calculate_prob_map(mask_path, self.n_classes, self.patch_s, self.vis_path)
            np.savez_compressed(self.cache_path / prob_name_full, prob_maps=pm_full)
            if self.prob_dx_factor != 1: 
                pm_downscaled = self._downscale_prob_map(pm_full, self.prob_dx_factor, self.prob_dx_func)
                np.savez_compressed(self.cache_path / prob_name_downscaled, prob_maps=pm_downscaled)
                del pm_full
                return pm_downscaled
            else:
                return pm_full

    @staticmethod
    def _postprocess_prob_map(prob_map, distancing=0.0):
        assert 0 <= distancing <= 1, 'invalid distancing coeff'
        res = []
        for m in prob_map:
            if m is not None:
                mbf = m.astype(np.float32)
                new_m = (mbf / np.max(mbf)) ** (1 - 0.9 * distancing)
                new_m /= np.sum(new_m)
                res.append(new_m)
            else:
                res.append(None)
        return res

    @staticmethod
    def _load_mask(p: Path):
        mask = np.array(Image.open(p)).astype(np.uint8)
        return mask if mask.ndim == 2 else mask[:, :, 0]

    @staticmethod
    def calculate_prob_map(mask_path: Path, n_classes: int, patch_s: int, vis_out_path: Path = None):
        mask_name = mask_path.stem
        mask = np.array(Image.open(mask_path)).astype(np.uint8)[:, :, 0]
        patch_prob_maps = []
        if vis_out_path is not None:
            vis_out_path.mkdir(exist_ok=True, parents=True)
        # --- iterate over all classes ---
        for cl in range(n_classes):
            m = np.where(mask == cl, 1, 0)
            class_absence = np.max(m) == 0
            if class_absence:
                patch_prob_maps.append(None)
            else:
                # --- get integral image with 1px padding of left upper corner ---
                integral = np.pad(integral_image(m), [(1, 0), (1, 0)], mode='constant')
                p = integral[:-patch_s, :-patch_s] + integral[patch_s:, patch_s:] - \
                    integral[:-patch_s, patch_s:] - integral[patch_s:, :-patch_s]
                p = np.pad(p, [(0, patch_s - 1), (0, patch_s - 1)], mode='constant')
                # --- normalize p ---
                min_p, max_p = np.min(p), np.max(p)
                p = (p - min_p) / (max_p - min_p)
                p = p / np.sum(p) # <----------------- ???
                # --- quantize p ---
                p = p.astype(np.float32)
                # --- visualize probability map if needed ---
                if vis_out_path is not None:
                    Image.fromarray(to_heat_map(p)).save(vis_out_path / f"{mask_name}_PPM_cl{cl}_{patch_s}.jpg")
                # --- append to maps list ---
                patch_prob_maps.append(p)
        return patch_prob_maps

    def get_patch(self):
        if self.mixin_random_every > 0 and self.patch_count % self.mixin_random_every == 0:
            return self.get_patch_random()
        else:
            return self._get_patch_balanced()

    def _get_patch_balanced(self):
        # --- choose class to generate patch ---
        t = time.perf_counter()
        if self.choose_strict_minority_class:
            # first way: choose minority class
            acc_classes = [(i, acc_px) for i, acc_px in enumerate(self.accumulated_px_per_class) if i not in self.missed_classes]
            cl = min(acc_classes, key=lambda p: p[1])[0]
        else:
            # second way: choose one from all minorities with some probability
            probs = [1 / (acc_px ** self.alpha) if acc_px > 0 else 1 for acc_px in self.accumulated_px_per_class]
            probs = [p if i not in self.missed_classes else 0 for i, p in enumerate(probs)]
            probs = np.array(probs) / sum(probs)
            cl = np.random.choice(self.n_classes, 1, p=probs)[0]
        self._t_class_select += time.perf_counter() - t
        # --- choose image to extract patch from ---
        t = time.perf_counter()
        img_idx = np.random.choice(self.n_imgs, 1, p=self.image_weights[cl])[0]
        img = self.imgs[img_idx]
        prob_map = self.prob_maps[img_idx][cl]
        self._t_img_select += time.perf_counter() - t
        # --- extract patch ---
        t = time.perf_counter()
        
        pos = np.random.choice(prob_map.size, 1, p=prob_map.flatten())[0]
        y = pos // prob_map.shape[1]
        x = pos % prob_map.shape[1]

        if self.prob_dx_factor > 1:
            y_shift = np.random.randint(self.prob_dx_factor)
            x_shift = np.random.randint(self.prob_dx_factor)
            y = y * self.prob_dx_factor + y_shift
            x = x * self.prob_dx_factor + x_shift
            y = min(img.shape[0] - self.patch_s, y)
            x = min(img.shape[1] - self.patch_s, x)
        
        patch_img = img[y : y + self.patch_s, x : x + self.patch_s]
        patch_mask = self.masks[img_idx][y : y + self.patch_s, x : x + self.patch_s]
        self._t_patch_extract += time.perf_counter() - t
        # --- update accumulated pixels ---
        t = time.perf_counter()
        self._update_accumulators(img_idx, y, x, patch_mask, cl)
        self._t_update_accumulators += time.perf_counter() - t
        return patch_img, patch_mask, cl
    
    def get_patch_random(self, update_accumulators=True):
        img_idx = np.random.randint(self.n_imgs)
        img = self.imgs[img_idx]
        mask = self.masks[img_idx]
        y = np.random.randint(img.shape[0] - self.patch_s)
        x = np.random.randint(img.shape[1] - self.patch_s)
        patch_img = img[y : y + self.patch_s, x : x + self.patch_s]
        patch_mask = mask[y : y + self.patch_s, x : x + self.patch_s]
        if update_accumulators:
            self._update_accumulators(img_idx, y, x, patch_mask)
        return patch_img, patch_mask, None

    def _update_accumulators(self, img_idx, y, x, patch_mask, cl=-1):
        for i in range(self.n_classes):
            self.accumulated_px_per_class[i] += np.count_nonzero(patch_mask == i)
        self.accumulated_image_maps[img_idx][y : y + self.patch_s, x : x + self.patch_s] += 1
        if cl != -1:
            self.accumulated_patches[cl][img_idx] += 1
        self.accumulated_patches_per_image[img_idx] += 1
        self.patch_count += 1

    def print_accumulators_info(self):
        print('### begin of accumulated info ###')
        acc_sum = np.sum(self.accumulated_px_per_class)
        print(f'pixels percentage accumulated per class: {[np.round(a / acc_sum, 3) for a in self.accumulated_px_per_class]}')
        for cl in range(self.n_classes):
            per_image_info = ', '.join([f'i{i}: {self.accumulated_patches[cl][i]}' for i in range(self.n_imgs)])
            print(f'\t patches extracted for class {cl}: {per_image_info}')
        patches_per_image = ', '.join([f'i{i}: {self.accumulated_patches_per_image[i]}' for i in range(self.n_imgs)])
        print(f'patches extracted for image: {patches_per_image}')
        print('### end of accumulated info ###')

    def vis_accumulators(self, iteration: int):
        self.vis_path.mkdir(exist_ok=True, parents=True)
        for i, m in enumerate(tqdm(self.accumulated_image_maps, 'visualizing selected pixels maps')):
            m2 = m.astype(np.float)
            m2 = m2 / np.max(m2) if np.max(m2) > 0 else m2
            p = to_heat_map(m2)
            if self.vis_path is not None:
                Image.fromarray(p).save(self.vis_path / f"selected_{iteration}_{i + 1}.jpg")

    def get_class_weights(self, remove_missed_classes):
        print('\t Calculating class weights...')
        weights = [0] * self.n_classes
        for mask in self.masks:
            for cl in range(self.n_classes):
                weights[cl] += np.count_nonzero(mask == cl)
        if remove_missed_classes:
            weights = [w for w in weights if w > 0]
        s = sum(weights)
        weights = [w / s for w in weights]
        return weights

    def benchmark(self, num_patches=100):
        self._t_class_select = 0
        self._t_img_select = 0
        self._t_patch_extract = 0
        self._t_update_accumulators = 0
        start_time = time.perf_counter()
        for _ in range(num_patches):
            self.get_patch()
        execution_time = time.perf_counter() - start_time
        print(f'Execution time: {execution_time}, {execution_time / num_patches} per iteration')
        print(f'\t class select time: {self._t_class_select}')
        print(f'\t image select time: {self._t_img_select}')
        print(f'\t patch extract time: {self._t_patch_extract}')
        print(f'\t update accums time: {self._t_update_accumulators}')

    def test_extraction_with_visualization(self, epochs, steps):
        for i in range(epochs):
            for _ in tqdm(range(steps), 'extracting patches'):
                self.get_patch()
            self.print_accumulators_info()
            self.vis_accumulators(i)


class SimpleBatchGenerator:

    def __init__(self, patch_generator, batch_s, n_classes, squeeze_mask, squeeze_mappings=None, augment=True) -> None:
        self.patch_generator = patch_generator
        self.batch_s = batch_s
        self.n_classes = n_classes if not squeeze_mask else len(squeeze_mappings)
        self.squeeze_mask = squeeze_mask
        self.mappings = squeeze_mappings
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
                mask = squeeze_mask(mask, self.mappings)
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
