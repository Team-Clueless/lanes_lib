import cv2
import numpy as np
from imutils import paths
import imutils


def start_end(iter, img_ht, img_wd):
    x1, x2 = 0, img_wd
    y = img_ht - (iter * 18 * img_ht / (39 * 39))

    return (x1, y, x2, y)


def largest(arr, n):
    if len(arr) < 1:
        mx = 0
    else:
        mx = cv2.contourArea(arr[0])
    
    for i in range(1, n):
        if cv2.contourArea(arr[i]) > mx:
            mx = cv2.contourArea(arr[i])
        
    return mx


def processImage(im):
    x1, y1 = 0, int(im.shape[0] / 1.8)
    y2, x2, _ = im.shape
    roi =  im[y1:y2, x1:x2]

    ##########################################################################
    # this section is for finding the right points for perspective transform # 
    ##########################################################################
    # cv2.circle(roi, (185, 20), 8, (0, 255, 255), -1) # topleft
    # cv2.circle(roi, (530, 20), 8, (0, 255, 255), -1) # topright
    # cv2.circle(roi, (0, 150), 8, (0, 255, 255), -1) # bottomleft
    # cv2.circle(roi, (800, 150), 8, (0, 255, 255), -1) # bottomright
    # return roi
    ##########################################################################

    # convert to grayscale
    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # gaussian blur to reduce noises
    blurred = cv2.GaussianBlur(gray_image, (7,7), cv2.BORDER_DEFAULT)

    # binary threshold to separate out lanes
    _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)

    # set up points for perspective transform to get a bird's eye view
    pts1 = np.float32([[185, 20], [530, 20], [0, 150], [800, 150]])
    pts2 = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])

    # get transform matrix
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # change perspective
    birds_eye = cv2.warpPerspective(thresh, matrix, (600, 600))

    return birds_eye


def getLanes(image):
    img_ht, img_wd, _ = image.shape

    upper_left = (0, 25 * img_ht // 40)
    bottom_right = (img_wd, img_ht)

    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray_scale, (7, 7), cv2.BORDER_DEFAULT)

    _, thresh = cv2.threshold(blur, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    diff_ht, diff_wd = (3 * img_ht // 20, img_wd // 2)
    
    inc_ht, inc_wd = diff_ht // 4, diff_wd // 4

    bottom_ht = 16 * img_ht // 20

    arr = [0 for _ in range(4)]

    for i in range(1, 5):
        _, contours, _ = cv2.findContours(thresh[bottom_ht-i*inc_ht:bottom_ht-(i-1)*inc_ht,img_wd/2:img_wd],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(image[bottom_ht-i*inc_ht:bottom_ht-(i-1)*inc_ht,img_wd/2:img_wd] ,contours,-1,(0,255,0),3)

        area = largest(contours, len(contours))

        arr[4-i] = area / inc_wd * 3
    
    return image, arr