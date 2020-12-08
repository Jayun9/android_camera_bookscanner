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

    def capture(self):
        self.iscaptured = True
        refPt = crop.crop(self.capture_img)
        self.input_img = self.capture_img[refPt[0][1]: refPt[1][1],
                                          refPt[0][0]: refPt[1][0]]

    def stream(self):
        cap = cv.VideoCapture(self.url)
        while(True):
            _, frame = cap.read()
            if frame is not None:
                cv.imshow('frame', frame)
            q = cv.waitKey(1)
            if q == ord('q'):
                break
            if self.iscaptured:
                self.capture_img = frame
                self.iscaptured = False
