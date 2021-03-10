import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform.integral import integral_image
from pathlib import Path


def to_heat_map(img, name='jet'):
    assert img.ndim == 2, 'shape {} is unsupported'.format(img.shape)
    assert (np.min(img) >= 0.0) and (np.max(img) <= 1.0), 'invalid range {} - {}'.format(np.min(img), np.max(img))
    cmap = plt.get_cmap(name)
    heat_img = cmap(img)[..., 0:3]
    return (heat_img * 255).astype(np.uint8)


class AutoBalancedPatchGenerator:

    def __init__(self, img_dir_path: Path, mask_dir_path: Path, cache_path: Path, patch_size: int, n_classes: int,
                 distancing, prob_capacity):
        self.image_dir_path = img_dir_path
        self.mask_dir_path = mask_dir_path
        self.cache_path = cache_path
        cache_path.mkdir(parents=True, exist_ok=True)
        self.patch_s = patch_size
        self.n_classes = n_classes
        self.distancing = distancing
        self.prob_capacity = prob_capacity
        # --- perform initialization ---
        print('Initializing patch generator...')
        self.img_paths = list(img_dir_path.iterdir())
        self.mask_paths = list(mask_dir_path.iterdir())
        assert len(self.img_paths) == len(self.mask_paths)
        # --- load all images, masks and prob maps ---
        self.imgs = self._load_imgs()
        self.masks = self._load_masks()
        self.prob_maps = self._load_maps()
        self.n_imgs = len(self.mask_paths)
        # --- calculate weights of images ---
        self.image_weights = self._calc_image_weights(self.masks)
        self.missed_classes = self._get_missed_classes(self.image_weights)
        print(f'\t missed classes: {self.missed_classes}')
        self.accumulated_px = [0] * n_classes
        print('Initializing patch generator finished.')

    def _load_imgs(self):
        print('\t Loading images...')
        return [np.array(Image.open(p)).astype(np.float32) / 256 for p in self.img_paths]
    
    def _load_masks(self):
        print('\t Loading masks...')
        return [self._load_mask(p) for p in self.mask_paths]
    
    def _load_maps(self):
        print('\t Loading probability maps...')
        maps = [self._get_prob_map(p) for p in self.mask_paths]
        # print('\t Posprocessing probability maps...')
        # maps = [self._postprocess_prob_map(pm, distancing=self.distancing) for pm in maps]
        return maps

    def _calc_image_weights(self, masks):
        print('\t Calculating weights of images...')
        image_weights = [None] * self.n_classes
        for cl in range(self.n_classes):
            px_sum = [np.sum(np.where(mask == cl, 1, 0)) for mask in masks]
            s = sum(px_sum)
            image_weights[cl] = px_sum / s if s > 0 else [0] * len(masks)
        return image_weights

    def _get_missed_classes(self, image_weights):
        missed = []
        for cl in range(self.n_classes):
            if sum(image_weights[cl]) == 0:
                missed.append(cl)
        return missed

    def _get_prob_map(self, mask_path: Path):
        prob_name = f'{mask_path.stem}_{self.patch_s}_{self.prob_capacity}.npz'
        prob_path = self.cache_path / prob_name
        if prob_path.exists():
            return np.load(prob_path, allow_pickle=True)['prob_maps']
        else:
            print(f'\t\t Probability map for {mask_path} not found in cache. Calculating...')
            pp_map = self.calculate_prob_map(mask_path, self.n_classes, self.patch_s, self.prob_capacity, vis_out_path=None)
            if not self.cache_path.exists():
                self.cache_path.mkdir()
            np.savez_compressed(prob_path, prob_maps=pp_map)  
            return pp_map

    @staticmethod
    def _postprocess_prob_map(prob_map, distancing=0.0):
        assert 0 <= distancing <= 1, 'invalid distancing coeff'
        res = []
        for m in prob_map:
            if m is not None:
                mbf = m.astype(np.float32)
                new_m = (mbf / np.max(mbf)) ** (1 - 0.9 * distancing)
                new_m /= np.sum(new_m)
                res.append(new_m.astype(np.float16))
            else:
                res.append(None)
        return res

    @staticmethod
    def _load_mask(p: Path):
        mask = np.array(Image.open(p)).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask

    @staticmethod
    def calculate_prob_map(mask_path: Path, n_classes: int, patch_s: int, prob_capacity: int, vis_out_path: Path = None):
        assert prob_capacity in (32, 64)
        mask_name = mask_path.stem
        mask = np.array(Image.open(mask_path)).astype(np.uint8)[:, :, 0]
        patch_prob_maps = []
        if vis_out_path is not None:
            vis_out_path.mkdir(exist_ok=True)
        # --- iterate over all classes ---
        for cl in range(n_classes):
            # print(f'Processing class {cl}')
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
                # --- visualize probability map if needed ---
                if vis_out_path is not None:
                    Image.fromarray(to_heat_map(p)).save(vis_out_path / f"{mask_name}_PPM_cl{cl}_{patch_s}.jpg")
                p = p / np.sum(p)
                if prob_capacity == 32:
                    p = p.astype(np.float32)
                elif prob_capacity == 64:
                    p = p.astype(np.float64)
                patch_prob_maps.append(p)
        return patch_prob_maps


    def get_patch(self):
        acc_classes = [(i, acc_px) for i, acc_px in enumerate(self.accumulated_px) if i not in self.missed_classes]
        cl = min(acc_classes, key=lambda p: p[1])[0]
        # print(f'\t\t choose class {cl}')
        img_idx = np.random.choice(self.n_imgs, 1, p=self.image_weights[cl])[0]
        # print(f'image idx: {img_idx}')
        prob_map = self.prob_maps[img_idx][cl]
        # print(f'prob_map: {prob_map.shape}')
        pos = np.random.choice(prob_map.size, 1, p=prob_map.flatten())[0]
        y = pos // prob_map.shape[1]
        x = pos % prob_map.shape[1]
        # print(f'x={x}, y={y}')
        patch_img = self.imgs[img_idx][y : y + self.patch_s, x : x + self.patch_s]
        patch_mask = self.masks[img_idx][y : y + self.patch_s, x : x + self.patch_s]
        # --- update accumulated pixels ---
        for i in range(self.n_classes):
            self.accumulated_px[i] += np.count_nonzero(patch_mask == i)
        return patch_img, patch_mask, cl

    def get_accumulated_distribution(self):
        acc_sum = sum(self.accumulated_px)
        return [np.round(a / acc_sum, 3) for a in self.accumulated_px]

    def precalc_patches(self, out_path: Path, n=1000, print_distribution_every=100):
        out_path.mkdir(exist_ok=True)
        for i in range(1, n + 1):
            patch_img, patch_mask, target_class = self.get_patch()
            name = f'patch_{i:07d}_cl_{target_class}'
            img_path = out_path / (name + '.png')
            mask_path = out_path / (name + '_m.png')
            Image.fromarray(patch_img).save(img_path)
            Image.fromarray(patch_mask).save(mask_path)
            if i % print_distribution_every == 0:
                print(f'{i} patches generated. Accum: {self.get_accumulated_distribution()}')

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


# pg = AutoBalancedPatchGenerator(
#     Path('c:\\dev\\#data\\LumenStone\\S1\\v1\\imgs\\train\\'),
#     Path('c:\\dev\\#data\\LumenStone\\S1\\v1\\masks\\train\\'),
#     Path('.\\maps\\'),
#     256, n_classes=13, distancing=0.0, prob_capacity=32)


# pg.precalc_patches(Path('.\\patches\\'), 100000)


# pg = AutoBalancedPatchGenerator(
#     Path('c:\\dev\\#data\\LumenStone\\S1\\v1\\imgs\\train\\'),
#     Path('c:\\dev\\#data\\LumenStone\\S1\\v1\\masks\\train\\'),
#     Path('.\\maps\\'),
#     544, n_classes=13, distancing=0.0, prob_capacity=32)


# pg.precalc_patches(Path('.\\patches_544\\'), 1000)

