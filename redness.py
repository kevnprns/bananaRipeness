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

    # final = cv2.drawContours(img, contourList, -1, (255,0, 0), 3)

    mask = np.zeros(img_rgb.shape,np.uint8)
    new_image = cv2.drawContours(mask,contourList,-1,255,-1,)
    new_image = cv2.bitwise_and(img_rgb,img_rgb,mask=mask)

    cv2.imwrite(writeFname, new_image);

def laplacianFilter(filename):
    img = cv2.imread(filename,0)
    laplacian = img.copy()

    cv2.Laplacian(laplacian,cv2.CV_64F)
    cv2.imwrite("laplacian.jpg", laplacian);

def getLabPercentages(img, height, width):
    bananaPixels = brownPixels = yellowPixels = greenPixels = 0.0
    for i in range(height):
        for j in range(width):
            l,a,b = img[i][j]
            a = a - 128
            b = b - 128
            l = (l * 100)/255
            if l < 100 and l > 0:
                if a >  15 and b < 58 and l > 19:
                    brownPixels = brownPixels + 1
                elif a < 18 and b > 47 and a > -7:
                    yellowPixels = yellowPixels + 1
                elif a < -7:
                    greenPixels = greenPixels + 1
                bananaPixels = bananaPixels + 1
    return yellowPixels/bananaPixels * 100, greenPixels/bananaPixels * 100, brownPixels/bananaPixels * 100, 

def analyzeImages():
    images = ["banana1.jpg", "banana2.jpg", "banana3.jpeg", "banana4.jpeg"]
    for image in images:
        imageName = image.split('.')
        filename = "images/threshold/" + imageName[0] + "_threshold." + imageName[1]
        print(filename)
        img = cv2.imread(filename)
        height, width = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB) 
        print(getLabPercentages(lab, height, width))

def convertToRedness(filename, writeFname):

    print ("\tReading file")
    redCh, greenCh, blueCh = imread_colour(filename)

    rgbIm = cv2.imread(filename)

    hsvIm = cv2.cvtColor(rgbIm, cv2.COLOR_RGB2HSV)

    hsvIm = cv2.blur(hsvIm,(5,5))

    h, s, v = cv2.split(hsvIm)

    print h

    yellow = (40, 90, 30)
    green = (120, 255, 255)

    mask = cv2.inRange(hsvIm, yellow, green)

    result = cv2.bitwise_and(rgbIm, rgbIm, mask=mask)



    # plt.subplot(1, 2, 1)
    # plt.imshow(mask, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(result)
    # plt.show()


    # newImage = np.zeros(redCh.shape,dtype=np.int16)
    #
    # newImageBlue = redCh * 0 ; # orange filter
    # newImageGreen = redCh * 0 ; # orange filter
    #
    # print ("\tConverting to redness")
    # newImage = ((3.5 * redCh) + (2 * greenCh)) - ((2 * greenCh) + blueCh); # orange included filter
    # # newImage = (3 * redCh) - (greenCh + blueCh); # red filter
    #
    # maximum = 0;
    # for i in range(0,redCh.shape[0]):
    #     newImage[i][newImage[i] < 300] = 0 # anything below 400 is set to 0.
    #     newMax = max(newImage[i])
    #
    #     if newMax > maximum:
    #         maximum = newMax
    #
    #
    # divider = 1
    # if maximum > 255: # if the values range higher than scale them between 0 and 255
    #     divider = maximum / 255
    #
    # newImage = newImage / divider
    #

    cv2.imwrite(writeFname,result)
    # # imwrite_colour("rednessColoured.jpg",newImage, newImageGreen, newImageBlue)


#imageList = ["banana1.jpg","banana2.jpg","banana3.jpeg","banana4.jpeg","banana5.jpg","banana6.jpg",
 #            "banana7.jpg","banana8.jpg","banana9.jpeg","banana10.jpg","banana11.jpg","banana12.jpg",
  #           "banana13.jpg","banana14.jpg","banana15.jpg","banana16.jpeg","banana17.jpeg"]
imageList = ["banana1.jpg", "banana2.jpg", "banana3.jpeg", "banana4.jpeg"]
            # ["citrus1.jpg","citrus2.jpg","citrus3.jpg","citrus4.jpg","citrus5.jpg","citrus6.jpg","citrus7.jpg","citrus8.jpg",
                        # "tomato1.jpg","tomato2.jpg","tomato3.jpg","tomato4.jpg","tomato6.jpg","tomato7.jpg","tomato8.jpg",
                        # ]

for imagePath in imageList:
    imageName = imagePath.split('.')
    print(imageName)
    originalFname = "images/fruit_Images/" + imageName[0] + "." + imageName[1]
    thresholdFname = "images/threshold/" + imageName[0] + "_threshold." + imageName[1]
    processedFname = "images/processed/" + imageName[0] + "_processed." + imageName[1]
    circledFname = "images/circled/" + imageName[0] + "_circled." + imageName[1]

    print("Beggining fruit detection for " + imageName[0] + "\n")

    convertToRedness(originalFname, thresholdFname);
    print("Finished Redness Conversion")
    fruitRecognition(thresholdFname, originalFname, processedFname);
analyzeImages()
    # print("Finished Fruit Recognition")


    # circleRecognition(processedFname, circledFname);
    # print("Finished Circle Detection\n\n")
