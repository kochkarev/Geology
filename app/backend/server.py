import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import label


active_anno_img: np.ndarray = None

class_indices = list(range(1, 17))


def send_string(s):
    print(json.dumps({'type': 'string', 'content': s}))
    sys.stdout.flush()


def send_signal(s):
    print(json.dumps({'type': 'sig', 'val': s}))
    sys.stdout.flush()

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


def send_array(img: np.ndarray, ext_type=None, optional=None):
    d = {'type': 'array', 'shape': img.shape}#, 'width': img.shape[1], 'height': img.shape[0]}
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


def bbox(img, v):
    rows = np.any(img == v, axis=1)
    cols = np.any(img == v, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    mask = img[rmin: rmax + 1, cmin: cmax + 1] // v
    return int(rmin), int(cmin), mask


def create_inst_anno(anno_path: str, id: int):
    global active_anno_img
    active_anno_img = np.array(Image.open(anno_path))[:, :, 0]
    send_string(f'annotation updated to {anno_path}, shape: {active_anno_img.shape}')
    inst_map = np.zeros(active_anno_img.shape[:2] + (3,), dtype=np.uint8)
    iid = 0
    for ci in class_indices:
        class_anno = np.where(active_anno_img == ci, 1, 0)
        if np.max(class_anno > 0):
            labeled, n = label(class_anno)
            for i in range(1, n + 1):
                r, c, mask = bbox(labeled, i)
                send_array(mask, ext_type='inst', optional={'id': iid, 'class': 0, 'r': r, 'c': c, 'imgid': id})
                inst_map[labeled == i, :] = [iid % 256, iid // 256 % 256, iid // 256 //256]
                iid += 1
    send_string(f'inst-map: {inst_map.shape}, {np.min(inst_map[:])}-{np.max(inst_map[:])}')
    send_array(inst_map, ext_type='inst-map', optional={'imgid': id})
    send_signal(f'A{id}')


while True:
    header = json.loads(input())
    command = header['type']
    if command == 'ping':
        send_string('pong')
    elif command == 'ping_image':
        w, h = 4, 4
        d = np.zeros([w, h, 3], dtype=np.uint8)
        for i in range(w):
            for j in range(h):
                d[i, j] = w * i + j
        send_array(d)
    elif command == 'stop':
        pass
    elif command == 'get-annotation':
        create_inst_anno(header['path'], int(header['id']))
    elif command == 'shutdown':
        break
