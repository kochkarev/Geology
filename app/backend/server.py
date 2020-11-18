import json
import os
import sys

import numpy as np
from PIL import Image


def send_string(s):
    print(json.dumps({'type': 'string', 'content': s}))
    sys.stdout.flush()


# def img_crop_to_layers(data: np.ndarray, n_layers):
#     q = 2 ** n_layers
#     h, w = data.shape[:2]
#     h = h // q * q
#     w = w // q * q
#     return data[:h, :w, ...]


def read_image_with_anno(header):
    w, h = int(header['width']), int(header['height'])
    img_path = header['image_path']
    support_level = header['support']
    step = header['step']
    anno = np.frombuffer(sys.stdin.buffer.read(w * h), dtype=np.uint8, count=w*h)
    anno = np.reshape(anno, [h, w])
    image = np.array(Image.open(img_path), dtype=np.uint8)
    return image, anno, support_level, step


def send_image(img: np.ndarray):
    print(json.dumps({'type': 'image', 'width': img.shape[1], 'height': img.shape[0]}))
    sys.stdout.flush()
    with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as stdout:
        stdout.write(img.tobytes())
    print()
    sys.stdout.flush()


while True:
    header = json.loads(input())
    command = header['type']
    if command == 'ping':
        send_string('pong')
    elif command == 'ping_image':
        w, h = 9000, 9000
        d = np.zeros([w, h], dtype=np.uint8)
        send_image(d)
    elif command == 'ping_coord':
        x = int(header['x'])
        y = int(header['y'])
        send_string(f'{x}-{y}')
        # send_string('ping_coord')
    elif command == 'stop':
        pass
    elif command == 'shutdown':
        break
