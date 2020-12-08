import numpy as np
from cv2 import cv2 as cv

def resize_img(src):
    max_x = 1280
    max_y = 700
    height, width = src.shape[:2]
    scale_x = float(width)/max_x
    scale_y = float(height)/max_y
    scale = int(np.ceil(max(scale_x, scale_y)))
    if scale > 1.0:
        inv_scl = 1.0/scale
        img = cv.resize(src, (0,0), None, inv_scl, inv_scl, cv.INTER_AREA)
    else:
        img = src
    return img

def run(inputimage):
    img = inputimage
    small = resize_img(img)
