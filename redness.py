#!/usr/local/bin/python2

from imageIO import *
# from imenh_lib import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# uses open cv contour to find objects in the scene and
# then eliminates the smaller and insignificant objects dependent on their ratio to the window.

def fruitRecognition(filename, originalFname, writeFname):
    print ("in canny")
    img_rgb = cv2.imread(originalFname,0)
    img = cv2.imread(filename)
    img = cv2.medianBlur(img,7)
    edges = cv2.Canny(img,215,255)

    height, width, channels = img.shape
    imageArea = width * height
    areaThreshold = imageArea / 1100

    edges = cv2.convertScaleAbs(edges)

    kernel = np.ones((3,3), np.uint8)
    dilated_image = cv2.dilate(edges,kernel,iterations=1)

    contours, _  = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours= sorted(contours, key = cv2.contourArea, reverse = True)

    contourList = []

    for c in contours:
        cArea = cv2.contourArea(c, True);

        if cArea < 0:
            if abs(cArea) > areaThreshold:
                contourList.append(c)

    final = cv2.drawContours(img, contourList, -1, (255,0, 0), 3)

    mask = np.zeros(img_rgb.shape,np.uint8)
    new_image = cv2.drawContours(mask,contourList,-1,255,-1,)
    new_image = cv2.bitwise_and(img_rgb,img_rgb,mask=mask)

    cv2.imwrite(writeFname, new_image);

def subtractBackground(img):

    fgbg = cv2.createBackgroundSubtractorMOG2()


    fgmask = fgbg.apply(img)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('frame',fgmask)


def convertToRedness(filename, writeFname):

    print ("\tReading file")
    redCh, greenCh, blueCh = imread_colour(filename)

    rgbIm = cv2.imread(filename)

    hsvIm = cv2.cvtColor(rgbIm, cv2.COLOR_RGB2HSV)

    hsvIm = cv2.blur(hsvIm,(5,5))

    h, s, v = cv2.split(hsvIm)

    print h

    yellow = (40, 100, 30)
    green = (120, 255, 255)

    # Range for greens
    lowerGreen = np.array([90,90,70])
    upperGreen = np.array([130,255,255])
    mask1 = cv2.inRange(hsvIm, lowerGreen, upperGreen)

    # Range for yellows
    lowerYellow = np.array([40,170,100])
    upperYellow = np.array([90,255,255])
    mask2 = cv2.inRange(hsvIm, lowerYellow, upperYellow)

    # mask = cv2.inRange(hsvIm, yellow, green)
    mask = mask1+mask2

    mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    #creating an inverted mask to segment out the cloth from the frame
    mask2 = cv2.bitwise_not(mask1)
    #Segmenting the cloth out of the frame using bitwise and with the inverted mask
    result = cv2.bitwise_and(rgbIm,rgbIm,mask=mask2)



    # result = cv2.bitwise_and(rgbIm, rgbIm, mask=mask)
    # subtractBackground(result)

    # edges = cv2.Canny(result,215,255)
    #
    # height, width, channels = result.shape
    # imageArea = width * height
    # areaThreshold = imageArea / 1100
    #
    # edges = cv2.convertScaleAbs(edges)
    #
    # kernel = np.ones((3,3), np.uint8)
    # dilated_image = cv2.dilate(edges,kernel,iterations=1)
    #
    # contours, _  = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contours= sorted(contours, key = cv2.contourArea, reverse = True)

    cv2.imwrite(writeFname,result)

imageList = ["banana1.jpg","banana2.jpg","banana3.jpeg","banana4.jpeg","banana5.jpg","banana6.jpg",
             "banana7.jpg","banana8.jpg","banana9.jpeg","banana10.jpg","banana11.jpg","banana12.jpg",
             "banana13.jpg","banana14.jpg","banana15.jpg","banana16.jpeg","banana17.jpeg"]


for imagePath in imageList:
    imageName = imagePath.split('.')
    print(imageName)
    originalFname = "images/fruit_Images/" + imageName[0] + "." + imageName[1]
    thresholdFname = "images/threshold/" + imageName[0] + "_threshold." + imageName[1]
    processedFname = "images/processed/" + imageName[0] + "_processed." + imageName[1]
    circledFname = "images/circled/" + imageName[0] + "_circled." + imageName[1]

    print("Beggining fruit detection for " + imageName[0] + "\n")

    convertToRedness(originalFname, thresholdFname);
    print("Finished Color Conversion")
    fruitRecognition(thresholdFname, originalFname, processedFname);
    # print("Finished Fruit Recognition")

cv2.destroyAllWindows()
