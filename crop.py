from cv2 import cv2 as cv

refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	global refPt, cropping
	if event == cv.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
        
	elif event == cv.EVENT_LBUTTONUP:
		refPt.append((x, y))
		cropping = False
		cv.rectangle(param, refPt[0], refPt[1], (0, 255, 0), 2)
		cv.imshow("image", param)
    

def crop(image):
    clone = image.copy()
    cv.namedWindow("image")
    cv.setMouseCallback("image", click_and_crop, param=image)
    while True:
        cv.imshow("image", image)
        key = cv.waitKey(1) & 0xFF
        if key == ord("r"):
            image = clone.copy()

        elif key == ord("c"):
            if len(refPt) == 2:
                roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                cv.imshow("ROI", roi)
                cv.waitKey(0)
        elif key == ord("q"):
            break
    cv.destroyAllWindows()
    return refPt
