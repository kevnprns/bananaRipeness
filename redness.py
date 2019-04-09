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

def fruitRecognition(originalFname, writeFname):
    print ("in canny")
    # img_rgb = cv2.imread(originalFname,0)
    img_rgb = cv2.imread(originalFname)
    img = cv2.imread(originalFname)
    # img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,7)

    edges = cv2.Canny(img,15,50,3,L2gradient=True)
    # edges = cv2.Canny(img, 10, 300, 3, L2gradient=True)

    height, width, channels = img.shape
    imageArea = width * height
    areaThreshold = imageArea * (20/100)

    edges = cv2.convertScaleAbs(edges)


    kernel = np.ones((3,3), np.uint8)
    dilated_image = cv2.dilate(edges,kernel,iterations=2)

    # cv2.imwrite(writeFname, dilated_image);

    contours, _  = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours= sorted(contours, key = cv2.contourArea, reverse = True)

    contourList = []

    for c in contours:
        cArea = cv2.contourArea(c, True);

        if cArea < 0:
            if abs(cArea) > areaThreshold:
                contourList.append(c)

    # cv2.imwrite(writeFname, dilated_image);


    mask = np.zeros(img_rgb.shape[:2],dtype=np.uint8)
    mask = cv2.drawContours(mask,contourList,-1,255,-1)
    # mask.astype(np.int8)
    # img_rgb.astype(np.int8)
    new_image = cv2.bitwise_and(img_rgb,img_rgb,mask=mask)

    # new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2RGB)

    cv2.imwrite(writeFname, new_image);

def subtractBackground(img):

    fgbg = cv2.createBackgroundSubtractorMOG2()


    fgmask = fgbg.apply(img)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('frame',fgmask)

def getLabPercentages(img, height, width):

    bananaPixels = brownPixels = yellowPixels = greenPixels = 0.0

    for i in range(height):
        for j in range(width):
            l,a,b = img[i][j]

            a = a - 128
            b = b - 128
            l = (l * 100)/255

            if l < 100 and l > 0:
                # if i==339 and j==224:
                #     print("{0} {1} {2}".format( l, a, b))
                if a <= 9 and b > 50 and a >= -16:
                    yellowPixels = yellowPixels + 1
                elif a >  9 and b < 48.5 and l >= 10:
                    brownPixels = brownPixels + 1
                elif a <= -5:
                    greenPixels = greenPixels + 1
    bananaPixels = yellowPixels + brownPixels + greenPixels
    return greenPixels/bananaPixels * 100, yellowPixels/bananaPixels * 100, brownPixels/bananaPixels * 100,

def analyzeImages(imageList):
    # images = ["banana1.jpg","banana2.jpg","banana3.jpeg","banana4.jpeg"]
    for image in imageList:
        imageName = image.split('.')
        filename = "images/threshold/" + imageName[0] + "_threshold." + imageName[1]
        print "\nAnalysing: " + filename
        img = cv2.imread(filename)
        height, width = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        print(getLabPercentages(lab, height, width))

def convertToRedness(filename, writeFname):

    redCh, greenCh, blueCh = imread_colour(filename)

    rgbIm = cv2.imread(filename)

    hsvIm = cv2.cvtColor(rgbIm, cv2.COLOR_RGB2HSV)

    hsvIm = cv2.blur(hsvIm,(5,5))

    h, s, v = cv2.split(hsvIm)

    yellow = (40, 100, 40)
    green = (130, 255, 255)

    # Range for greens
    lowerGreen = np.array([75,100,40])
    upperGreen = np.array([130,255,255])
    mask1 = cv2.inRange(hsvIm, lowerGreen, upperGreen)

    # Range for yellows
    lowerYellow = np.array([40,30,100])
    upperYellow = np.array([75,255,255])
    mask2 = cv2.inRange(hsvIm, lowerYellow, upperYellow)

    # Range for yellows
    lowerBrown = np.array([30,0,0])
    upperBrown = np.array([130,255,140])
    mask3 = cv2.inRange(hsvIm, lowerBrown, upperBrown)

    # mask = cv2.inRange(hsvIm, yellow, green)
    mask = mask1+mask2 + mask3

    result = cv2.bitwise_and(rgbIm, rgbIm, mask=mask)

    cv2.imwrite(writeFname,result)



imageList = [
            "banana1.jpg","banana2.jpg","banana3.jpeg","banana4.jpeg","banana5.jpg",
            "banana6.jpg",
             # "banana7.jpg","banana8.jpg","banana9.jpeg","banana10.jpg","banana11.jpg","banana12.jpg",
             # "banana13.jpg","banana14.jpg","banana15.jpg","banana16.jpeg","banana17.jpeg"
             ]


for imagePath in imageList:
    imageName = imagePath.split('.')
    originalFname = "images/fruit_Images/" + imageName[0] + "." + imageName[1]
    thresholdFname = "images/threshold/" + imageName[0] + "_threshold." + imageName[1]
    processedFname = "images/preprocessed/" + imageName[0] + "_preprocessed." + imageName[1]
    circledFname = "images/circled/" + imageName[0] + "_circled." + imageName[1]

    print("Beggining banana threshold for " + imageName[0] + "\n")

    fruitRecognition(originalFname, processedFname);
    convertToRedness(processedFname, thresholdFname);

print("Finished threshold")

analyzeImages(imageList)
