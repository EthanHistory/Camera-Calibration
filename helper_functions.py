import os
import glob

import cv2 as cv


def chessboard_images(path='./images'):
    images_path = glob.glob(os.path.join(path, "*.jpg"))
    print(images_path)
    images = [cv.imread(fname) for fname in images_path]
    return images

def extract_corners(image, pattern_size, criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria).squeeze()
        return corners2
    return None