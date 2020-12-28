import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import label
import tensorflow as tf
from pathlib import Path
from utils import split_to_patches, combine_patches


def bbox(img, v):
    rows = np.any(img == v, axis=1)
    cols = np.any(img == v, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    mask = img[rmin: rmax + 1, cmin: cmax + 1]
    mask = np.where(mask == v, 1, 0)
    return int(rmin), int(cmin), mask.astype(np.uint8)


def read_img(anno_path: str, squeeze=True):
    img = np.array(Image.open(anno_path))
    if squeeze and img.ndim == 3:
        img = img[:, :, 0]
    return img


def remove_dublicate_labels(d: np.ndarray):
    # dublicated values:
    # Py/Mrc: 6, 13, 15
    # Shp: 8, 12
    d = np.where(d == 13, 6, d)
    d = np.where(d == 15, 6, d)
    d = np.where(d == 12, 8, d)
    return d

class Server:

    def __init__(self, n_classes, patch_s, batch_s, offset):
        self.class_indices = list(range(1, n_classes + 1))
        self.active_img: np.ndarray = None
        self.model = None
        self.patch_s = patch_s
        self.batch_s = batch_s
        self.offset = offset

    def send_string(self, s):
        print(json.dumps({'type': 'string', 'content': s}))
        sys.stdout.flush()

    def send_signal(self, s):
        print(json.dumps({'type': 'sig', 'val': s}))
        sys.stdout.flush()

    def send_array(self, img: np.ndarray, ext_type=None, optional=None):
        d = {'type': 'array', 'shape': img.shape}
        if ext_type is not None:
            d['ext'] = ext_type
            if optional is not None:
                d.update(optional)
        print(json.dumps(d))
        sys.stdout.flush()
        with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as stdout:
            stdout.write(img.tobytes())
        print()
        sys.stdout.flush()

    def create_inst_anno(self, anno_path: str, id: int, source: str, area_thresh):
        return self.create_inst_anno_img(read_img(anno_path), id, source, area_thresh)

    def create_inst_anno_img(self, img, id: int, source: str, area_thresh):
        self.send_string(f'creatig inst-map for image {id}')
        inst_map = np.zeros(img.shape[:2] + (3,), dtype=np.uint8)
        iid = 1
        inst_dropped = 0
        for ci in self.class_indices:
            class_anno = np.where(img == ci, 1, 0)
            if np.max(class_anno > 0):
                labeled, n = label(class_anno)
                for i in range(1, n + 1):
                    r, c, mask = bbox(labeled, i)
                    mask_area = np.sum(mask)
                    if mask_area < area_thresh:
                        inst_dropped += 1
                        continue
                    meta = {'src': source, 'id': iid, 'class': ci, 'y': r, 'x': c, 'imgid': id, 'area': str(mask_area)}
                    self.send_array(mask, ext_type='inst', optional=meta)
                    inst_map[labeled == i, :] = [iid // (256 * 256), (iid // 256) % 256, iid % 256]
                    iid += 1
        self.send_string(f'inst-map: {inst_map.shape}, instances: {iid-1}, dropped: {inst_dropped}, src: {source}')
        self.send_array(inst_map, ext_type='inst-map', optional={'src': source, 'imgid': id})
        if (source == 'GT'):
            self.send_signal(f'A{id}')
        elif (source == 'PR'):
            self.send_signal(f'B{id}')

    def load_model(self, name):
        model_path = Path('.\\backend\\models') / (name + '.hdf5')
        if not model_path.exists():
            return
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.send_string('Model loaded')
        self.send_signal('L1')

    def _predict_np(self, img):
        patches = split_to_patches(img, self.patch_s, self.offset, overlay=0.25)
        init_patch_len = len(patches)

        while (len(patches) % self.batch_s != 0):
            patches.append(patches[-1])
        p_patches = []

        for i in range(0, len(patches), self.batch_s):
            batch = np.stack(patches[i : i+self.batch_s])
            prediction = self.model.predict_on_batch(batch)
            for x in prediction:
                p_patches.append(x)
        
        p_patches = p_patches[:init_patch_len]
        result = combine_patches(p_patches, self.patch_s, self.offset, overlay=0.25, orig_shape=(img.shape[0], img.shape[1], p_patches[0].shape[2]))
        return result

    def post_proc_prediction(self, pred: np.ndarray, src_shape):
        h, w = src_shape[0: 2]
        res = np.zeros([h, w], dtype=np.uint8)
        sh_y = (h - pred.shape[0]) // 2
        sh_x = (w - pred.shape[1]) // 2
        a = np.argmax(pred, axis=2)
        res[sh_y : sh_y + a.shape[0], sh_x : sh_x + a.shape[1]] = a
        # tidying up labels
        res += 100
        res = np.where(res == 100, 0, res)  # 0 : "Other" -> 0 
        res = np.where(res == 101, 8, res) # 1 : "Sh" -> 8
        res = np.where(res == 102, 6, res)  # 2 : "PyMrc" -> 6
        res = np.where(res == 103, 2, res)  # 3 : "Gl" -> 2
        # convert mask to 3-channels
        res = np.stack([res, res, res], axis=2)
        return res, sh_x # offset should be equal at all 4 sides
    
    def predict(self, img_path: str, id: int, anno_path=None):
        img = np.array(Image.open(img_path)).astype(np.float32) / 255
        self.send_string(f'predicting for shape: {img.shape}')
        if self.model is not None:
            # predict segmentation
            prediction = self._predict_np(img)
            pp, offset = self.post_proc_prediction(prediction, img.shape)
            self.send_string(f'prediction: {img.shape} -> {pp.shape}')
            self.send_array(pp, 'pred', optional={'imgid': id, 'shape': pp.shape})
            # split to instances
            self.create_inst_anno_img(pp[:, :, 0], id, 'PR', area_thresh=10)
            # calculate error map if anno_path is not empty
            if anno_path:
                gt = read_img(anno_path, squeeze=False)
                gt = remove_dublicate_labels(gt)
                err = self.error_map(gt, pp, offset)
                self.send_string(f'error map: {err.shape}')
                self.send_array(err, 'err', optional={'imgid': id, 'shape': err.shape})
        else:
            self.send_string('model is not loaded')

    def error_map(self, gt, pred, offset):
        # error_map codes: 253 - unknown, 254 - correct, 255 - error
        gt_valid = gt[offset : -offset, offset : -offset, 0]
        pred_valid = pred[offset : -offset, offset : -offset, 0]
        err_valid = np.where(gt_valid == pred_valid, 254, 255)
        err = np.zeros(gt.shape[:2], dtype=np.uint8) + 253
        err[offset: -offset, offset: -offset] = err_valid
        return np.stack([err, err, err], axis=2)

    def _ping_image(self):
        w, h = 4, 4
        d = np.zeros([w, h, 3], dtype=np.uint8)
        for i in range(w):
            for j in range(h):
                d[i, j] = w * i + j
        self.send_array(d)

    def run(self):
        while True:
            msg = json.loads(input())
            command = msg['type']
            if command == 'ping':
                self.send_string('pong')
            elif command == 'ping_image':
                self._ping_image()
            elif command == 'stop':
                pass
            elif command == 'load-model':
                self.load_model(msg['name'])
            elif command == 'image-predict':
                gt_path = msg.get('gt-path', None)
                self.predict(msg['path'], int(msg['id']), gt_path)
            elif command == 'get-annotation':
                self.create_inst_anno(msg['path'], int(msg['id']), 'GT', area_thresh=10)
            elif command == 'shutdown':
                break


server = Server(n_classes=16, patch_s=256, batch_s=32, offset=8)
server.run()
