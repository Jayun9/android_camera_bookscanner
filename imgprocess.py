from cv2 import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import crop
import scann


class process():
    def __init__(self, url):
        self.img_number = 1
        self.url = url
        self.iscaptured = False
        self.capture_img = None
        self.input_img = None
        self.stream_stop = False

    def capture(self):
        self.iscaptured = True
        while self.input_img is None:
            continue
        captured_image = self.input_img.copy()
        self.capture_img = captured_image
        self.input_img = None
        print(captured_image.shape)
        widht = captured_image.shape[1]
        fx = 600/widht 
        fy = fx
        captured_image = cv.resize(captured_image, dsize=(0,0), fx=fx, fy=fy, interpolation=cv.INTER_AREA)
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
            if self.stream_stop:
                break
            if self.iscaptured:
                capture_img = origin
                refPt = crop.crop(capture_img)
                self.input_img = capture_img[refPt[0][1]: refPt[1][1],
                                                refPt[0][0]: refPt[1][0]]
                refPt = None
                self.iscaptured = False

    def run(self):
        self.result_img = scann.run(self.capture_img)
        width = self.result_img.shape[1]
        fx = 600 / width
        result_img = cv.resize(self.result_img, dsize=(0,0), fx=fx, fy=fx, interpolation=cv.INTER_LINEAR)
        return result_img

    def save(self, save_folder_name):
        save_folder = f'./{save_folder_name}'
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        filename = f'{save_folder}/{self.img_number}.jpg'
        cv.imwrite(filename, self.result_img)  

