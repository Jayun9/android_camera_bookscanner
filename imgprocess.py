from cv2 import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import crop
import scann


class process():
    def __init__(self, url):
        self.url = url
        self.iscaptured = False
        self.capture_img = None
        self.input_img = None

    def capture(self):
        self.iscaptured = True
        while self.input_img is None:
            continue
        captured_image = self.input_img.copy()
        self.capture_img = captured_image
        self.input_img = None
        return captured_image

    def stream(self):
        cap = cv.VideoCapture(self.url)
        while(True):
            _, frame = cap.read()
            if frame is not None:
                origin = frame.copy()
                re_frame = cv.resize(frame, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
                cv.imshow('frame', re_frame)
            q = cv.waitKey(1)
            if q == ord('q'):
                break
            if self.iscaptured:
                capture_img = origin
                refPt = crop.crop(capture_img)
                self.input_img = capture_img[refPt[0][1]: refPt[1][1],
                                                refPt[0][0]: refPt[1][0]]
                refPt = None
                self.iscaptured = False

    def run(self):
        scann.run(self.capture_img)
