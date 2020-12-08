from cv2 import cv2 as cv
import numpy as np

def stream():
    url = input()
    full_url = f'https://{url}/video'
    cap = cv.VideoCapture(full_url)
    while(True):
        ret, frame = cap.read()
        if frame is not None:
            cv.imshow('frame', frame)
            return frame
        q = cv.waitKey(1)
        if q == ord('q'):
            break
    cv.destroyAllWindows()
