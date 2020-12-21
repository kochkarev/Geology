import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import label


# def img_crop_to_layers(data: np.ndarray, n_layers):
#     q = 2 ** n_layers
#     h, w = data.shape[:2]
#     h = h // q * q
#     w = w // q * q
#     return data[:h, :w, ...]


# def read_image_with_anno(header):
#     w, h = int(header['width']), int(header['height'])
#     img_path = header['image_path']
#     support_level = header['support']
#     step = header['step']
#     anno = np.frombuffer(sys.stdin.buffer.read(w * h), dtype=np.uint8, count=w*h)
#     anno = np.reshape(anno, [h, w])
#     image = np.array(Image.open(img_path), dtype=np.uint8)
#     return image, anno, support_level, step

def bbox(img, v):
    rows = np.any(img == v, axis=1)
    cols = np.any(img == v, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    mask = img[rmin: rmax + 1, cmin: cmax + 1]
    mask = np.where(mask == v, 1, 0)
    return int(rmin), int(cmin), mask.astype(np.uint8)


class Server:

    def __init__(self, n_classes):
        self.class_indices = list(range(1, n_classes + 1))
        self.active_anno_img: np.ndarray = None

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

    def create_inst_anno(self, anno_path: str, id: int, area_thresh=10):
        self.active_anno_img = np.array(Image.open(anno_path))[:, :, 0]
        self.send_string(f'annotation updated to {anno_path}, shape: {self.active_anno_img.shape}')
        inst_map = np.zeros(self.active_anno_img.shape[:2] + (3,), dtype=np.uint8)
        iid = 1
        inst_dropped = 0
        for ci in self.class_indices:
            class_anno = np.where(self.active_anno_img == ci, 1, 0)
            if np.max(class_anno > 0):
                labeled, n = label(class_anno)
                for i in range(1, n + 1):
                    r, c, mask = bbox(labeled, i)
                    mask_area = np.sum(mask)
                    if mask_area < area_thresh:
                        inst_dropped += 1
                        continue
                    self.send_array(mask, ext_type='inst',optional={'id': iid, 'class': ci, 'y': r, 'x': c, 'imgid': id, 'area': str(mask_area)})
                    inst_map[labeled == i, :] = [iid % 256, iid // 256 % 256, iid // 256 //256]
                    iid += 1
        self.send_string(f'inst-map: {inst_map.shape}, instances: {iid-1}, dropped: {inst_dropped}')
        self.send_array(inst_map, ext_type='inst-map', optional={'imgid': id})
        self.send_signal(f'A{id}')

    def load_model(self, name, epoch=None):
        pass

    def run(self):
        while True:
            header = json.loads(input())
            command = header['type']
            if command == 'ping':
                self.send_string('pong')
            elif command == 'ping_image':
                w, h = 4, 4
                d = np.zeros([w, h, 3], dtype=np.uint8)
                for i in range(w):
                    for j in range(h):
                        d[i, j] = w * i + j
                self.send_array(d)
            elif command == 'stop':
                pass
            elif command == 'get-annotation':
                self.create_inst_anno(header['path'], int(header['id']))
            elif command == 'shutdown':
                break


server = Server(n_classes=16)
server.run()
