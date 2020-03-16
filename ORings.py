import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import math

#Parameters
num = int(sys.argv[1])
#Variable declarations
labelCounter = 0
colourArray = [40,80,120,160,200,240]
centerPoint = [0,0]
#For calculating average distance
outsideXValues = []
outsideYValues = []
insideXValues = []
insideYValues = []
#For labeling black values 
blackCounter = 1
#For calculating if it is faulty
averageWidth = 0
averageOutsideRadius = 0
averageInsideRadius = 0

#Apply threshold and create monocolour image
def threshold(img,thresh):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i,j] > thresh:
                img[i,j] = 255
            else:
                img[i,j] = 0
    return img

#Draw histogram
def imhist(img):
    hist = np.zeros(256)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            hist[img[i,j]]+=1
    return hist

#Get threshold
def getThresh(hist):
    maxval = max(hist)
    maxIndex = np.where(hist == maxval)
    thresh = maxIndex[0] - 75
    return thresh

#Binary morpholgy
def binaryMorphology(img, type, iterations):
    copyImg = img.copy()
    for x in range(0, iterations):
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if img[i,j] == 255 and type == "dilate": 
                    copyImg = dilate(i,j,copyImg)
                if img[i,j] == 0 and type == "expand": 
                    copyImg = expand(i,j,copyImg)
    return copyImg

#Expand
def expand(i,j,img):
    xi,xj = i-1,j-1
    for x in range(xi, i+2):
        for q in range(xj, j+2):
            if(0 < x < img.shape[0] and 0 < q < img.shape[1]):
                img[x,q] = 0
    return img

#Dilate
def dilate(i,j,img):
    xi,xj = i-1,j-1
    for x in range(xi, i+2):
        for q in range(xj, j+2):
            if(0 < x < img.shape[0] and 0 < q < img.shape[1]):
                img[x,q] = 255
    return img

#Searches for white pixels
def labeling(img):
    global labelCounter
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            if img[i,j] == 255:
               img = allConnected(i,j,img,255)
               labelCounter+=1
    return img
                
#Changes colour of pixel
def allConnected(i,j,img,color):
    global labelCounter
    img[i,j] = colourArray[labelCounter]
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if img[x,y] != 0 and checkNeighbours(x,y,img) == True:
                img[x,y] = colourArray[labelCounter]
    for x in range(img.shape[0] - 1, 0, -1):
        for y in range(img.shape[1] - 1, 0, -1):
            if img[x,y] == color and checkNeighbours(x,y,img) == True:
                img[x,y] = colourArray[labelCounter]
    for x in range(0, img.shape[0]):
        img[0,x] = colourArray[0]
    return img

#Checks all pixels around it
def checkNeighbours(i,j,img):
    xi,xj = i-1,j-1
    for x in range(xi, i+2):
        for q in range(xj, j+2):
            if(0 < x < img.shape[0] and 0 < q < img.shape[1]):
                if img[x,q] == colourArray[labelCounter]:
                    return True
    return False

#Check for connected black pixels
def connectedBlack(img):
    global blackCounter
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if img[x,y] == 0:
                getConnected(img, x, y)
                blackCounter += 1
    return img

#Label the black pixels
def getConnected(img, x, y):
    global blackCounter
    img[x,y] = blackCounter
    while checkForConnected(img):
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if img[i,j] == blackCounter:                
                    colourNeighbours(img, i,j, blackCounter)
    return img

#Colour black pixels
def colourNeighbours(img, i, j, blackCounter):
    xi,xj = i-1,j-1
    for x in range(xi, i+2):
        for q in range(xj, j+2):
            if(0 < x < img.shape[0] and 0 < q < img.shape[1]):
                if img[x,q] == 0:
                    img[x,q] = blackCounter
    return img

#Check if black pixels are connected
def checkForConnected(img):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i,j] == blackCounter:
                xi,xj = i-1,j-1
                for x in range(xi, i+2):
                    for q in range(xj, j+2):
                        if(0 < x < img.shape[0] and 0 < q < img.shape[1]):
                            if img[x,q] == 0:
                                return True
    return False

#Remove the outlier
def removeOutliers(img):
    surroundColour = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i,j] == 2:
                if surroundColour == 0:
                    surroundColour = getSurrounding(img, i, j)
                    img[i,j] = surroundColour
                else:
                    img[i,j] = surroundColour
    return img             
    
#Get the surrounding marker of the outlier
def getSurrounding(img, i, j):
    xi,xj = i-1,j-1
    for x in range(xi, i+2):
        for q in range(xj, j+2):
            if(0 < x < img.shape[0] and 0 < q < img.shape[1]):
                if img[x,q] != 2:
                    return img[x,q]

#Find center point
def findCenter(img):
    global centerPoint
    xTotal = 0
    yTotal = 0
    num = 0
    for x in range(1, img.shape[0]):
        for y in range(1, img.shape[1]):
            if img[x,y] == 1:
                xTotal += x
                yTotal += y
                num += 1
    centerPoint = [round(xTotal/num), round(yTotal/num)] 
    img = drawCenter(round(xTotal/num),round(yTotal/num),img)
    return img

#Draw the center point
def drawCenter(i,j,img):
    xi,xj = i-1,j-1
    for x in range(xi, i+1):
        for j in range(xj, j+1):
            img[x,j] = 255
    return img

#Outline the inside and outside of the circle    
def outlineBorders(img):
    global labelCounter
    for x in range(0, labelCounter):
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if img[i,j] == 1 and checkOutlineNeighbours(i,j,x,img):
                    img[i,j] = 255 - 1 - x
                    if img[i,j] == 254:
                        outsideXValues.append(i)
                        outsideYValues.append(j)
                    else:
                        insideXValues.append(i)
                        insideYValues.append(j)

#Chcek if pixels are actually on the border
def checkOutlineNeighbours(i,j,y,img):
    xi,xj = i-1,j-1
    for x in range(xi, i+2):
        for q in range(xj, j+2):
            if(0 < x < img.shape[0] and 0 < q < img.shape[1]):
                if(img[x][q] == colourArray[y]):
                    return True

#Get the average thickness of the ring
def getAverageThickness(outsideX, outsideY, insideX, insideY):
    global centerPoint
    global averageWidth
    global averageOutsideRadius
    global averageInsideRadius

    totalOutsideDist = 0
    totalInsideDist = 0
    for x in range(0, len(outsideX)):
        totalOutsideDist += getDistance(outsideX[x],outsideY[x],centerPoint[0],centerPoint[1])
    for x in range(0, len(insideX)):
        totalInsideDist += getDistance(insideX[x],insideY[x],centerPoint[0],centerPoint[1])

    averageOutsideRadius = totalOutsideDist/len(outsideX)
    averageInsideRadius = totalInsideDist/len(insideX)
    averageWidth = averageOutsideRadius - averageInsideRadius

#I got this function from a forum: https://community.esri.com/thread/158038
#Get distance between two points
def getDistance(x1,y1,x2,y2):  
     distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return distance  

#Checks if the band is broken
def isBroken(img):
    global labelCounter
    if(labelCounter == 1):
        return "Fail"
    elif(hasMissingChunk()):
        return "Fail"
    return "Pass"

#Checks if there is a chunk missing
def hasMissingChunk():
    global centerPoint
    global averageWidth
    global averageOutsideRadius
    global averageInsideRadius
    global outsideXValues
    global outsideYValues

    min = averageOutsideRadius - averageWidth/5.5

    for x in range(0, len(outsideXValues)):
        if not min <= getDistance(outsideXValues[x],outsideYValues[x],centerPoint[0],centerPoint[1]):
            return True
    return False

#Got this from a website. https://www.programiz.com/python-programming/methods/built-in/round
def addAnnotation(time, img):
    font = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,20)
    fontScale = 0.5
    fontColor = (255,255,255)
    lineType = 2

    cv.putText(img,str(round(time, 5)) + " seconds", 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)

    return img

#Before time
before = time.time()

#Read in the image
img = cv.imread('Orings/Oring' + str(num) + '.jpg',0)

#Draw histogram and apply thresholding
hist = imhist(img)
threshold(img,getThresh(hist))

#Expansion and Dilation
img = binaryMorphology(img,"expand",2)
img = binaryMorphology(img,"dilate",2)

#Get connected components
labeling(img)

if not labelCounter == 1:
    #Label black components
    connectedBlack(img)
    removeOutliers(img)

    #Get center and draw outlines
    findCenter(img)
    outlineBorders(img)

    getAverageThickness(outsideXValues, outsideYValues, insideXValues, insideYValues)
    #Stats
    #print("Average outside radius: " + str(averageOutsideRadius))
    #print("Average inside radius: " + str(averageInsideRadius))
    #print("Average width: " + str(averageWidth))

after = time.time()

addAnnotation(after-before, img)

#Show resulting image
cv.imshow(isBroken(img),img)

cv.waitKey(0)
cv.destroyAllWindows()