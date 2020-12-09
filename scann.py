from matplotlib.pyplot import axis
from page_dewarp import ContourInfo, get_contours, make_tight_mask
import numpy as np
from cv2 import cv2 as cv

TEXT_MIN_W = 15
TEXT_MIN_H = 2
TEXT_MIN_A = 2
TEXT_MAX_T = 10


def resize_img(src):
    max_x = 1280
    max_y = 700
    height, width = src.shape[:2]
    scale_x = float(width)/max_x
    scale_y = float(height)/max_y
    scale = int(np.ceil(max(scale_x, scale_y)))
    if scale > 1.0:
        inv_scl = 1.0/scale
        img = cv.resize(src, (0, 0), None, inv_scl, inv_scl, cv.INTER_AREA)
    else:
        img = src
    return img

def get_margin(re_img):
    height, width = re_img.shape[:2]
    xmin = 20
    ymin = 20
    xmax = width - xmin
    ymax = height - ymin
    page = np.zeros((height, width), dtype=np.uint8)
    cv.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)
    outline = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin],
    ])
    return page, outline

def get_mask(re_img, margin):
    re_img_gray = cv.cvtColor(re_img, cv.COLOR_BGR2GRAY)
    mask = cv.adaptiveThreshold(
        re_img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 55, 25)
    mask = cv.dilate(mask, np.ones((9, 1), dtype=np.uint8))
    mask = cv.erode(mask, np.ones((1, 3), dtype=np.uint8))
    return np.minimum(mask, margin)

def make_tight_mask(contour, xmin, ymin, width, height):
    tight_mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1,1,2))
    cv.drawContours(tight_mask, [tight_contour], 0, (1,1,1), -1)
    return tight_mask

def bold_mean_and_tangent(contour):
    moments = cv.moments(contour)
    area = moments['m00']
    mean_x = moments['m10']/ area
    mean_y = moments['m01']/ area
    moment_matrix = np.array([
        [moments['mu20'], moments['mu11']],
        [moments['mu11'], moments['mu02']]
    ]) / area
    _, svd_u, _ = cv.SVDecomp(moment_matrix)


class ContourInfo(object):
    def __init__(self, contour, rect, mask):
        self.contour = contour
        self.rect = rect
        self.mask = mask
        self.center, self.tangent = bold_mean_and_tangent(contour)

def get_contours(re_img, margin):
    mask = get_mask(re_img, margin)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_list = []
    for contour in contours:
        rect = cv.boundingRect(contour)
        xmin, ymin, width, height = rect
        if (width < TEXT_MIN_W or
            height < TEXT_MIN_H or
            width < TEXT_MIN_A*height):
            continue
        tight_mask = make_tight_mask(contour, xmin, ymin, width, height)
        if tight_mask.sum(axis=0).max() > TEXT_MAX_T:
            continue
        contours_list.append(ContourInfo(contour, rect, tight_mask))
    return contours_list


def run(inputimage):
    img = inputimage
    re_img = resize_img(img)
    margin, outline = get_margin(re_img)
    contours_information_list = get_contours(re_img, margin)
    return contours_information_list
