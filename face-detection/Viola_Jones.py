import numpy as np
import os
import cv2
import sys
from skimage import io
import Feature as f
from PIL import Image

# loading images
def loadImage(address):
    images = []
    for filename in os.listdir(address):
        if 'pgm' in filename:
            tempImage = np.asarray(Image.open(address+'/'+filename))
            if tempImage.max() == 0:
                images.append(tempImage)
            else:
                images.append(tempImage/tempImage.max())
    return images

# get integral images
def integralImage(image):
    integral = np.zeros((image.shape[0], image.shape[1]))
    temp = np.zeros((image.shape[0], image.shape[1]))
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            if x>=1 and y>=1:
                temp[y,x] = temp[y,x-1] + image[y,x]
                integral[y,x] = integral[y-1,x] + temp[y,x]
            elif y>=1:
                temp[y,x] = image[y, x]
                integral[y, x] = integral[y - 1, x] + temp[y, x]
            elif x>= 1:
                temp[y, x] = temp[y, x - 1] + image[y, x]
                integral[y, x] = temp[y, x]
            else:
                temp[y, x] = image[y, x]
                integral[y, x] = temp[y, x]
    return integral

# 2 Rectangle horizontal
def getAll2RectangleHorizontalFeatures(integralInmage,allfeature,initialWidth,initialHeight):
    maxHeight = max(initialHeight, 2)
    maxWidth = max(initialWidth,1)
    for x in range((integralInmage.shape)[0]-maxHeight):
        for y in range((integralInmage.shape)[1]-maxWidth):
            allfeature.append(f.Features('2RectangleHorizontal', (x, y), maxWidth,maxHeight,1,1))
            allfeature.append(f.Features('2RectangleHorizontal', (x, y), maxWidth,maxHeight,-1,1))

# 2 Rectangle vertical
def getAll2RectangleVerticalFeatures(integralInmage,allfeature,initialWidth,initialHeight):
    maxHeight = max(initialHeight, 1)
    maxWidth = max(initialWidth, 2)
    for x in range((integralInmage.shape)[0] - maxHeight):
        for y in range((integralInmage.shape)[1] - maxWidth):
            allfeature.append(f.Features('2RectangleVertical',(x,y), maxWidth,maxHeight,1,1))
            allfeature.append(f.Features('2RectangleVertical',(x,y), maxWidth,maxHeight,-1,1))

# 3 Rectangle vertical
def getAll3RectangleVerticalFeatures(integralInmage,allfeature,initialWidth,initialHeight):
    maxHeight = max(initialHeight, 1)
    maxWidth = max(initialWidth, 3)
    for x in range((integralInmage.shape)[0] - maxHeight):
        for y in range((integralInmage.shape)[1] - maxWidth):
            allfeature.append(f.Features('3RectangleVertical',(x,y),maxWidth,maxHeight,1,1))
            allfeature.append(f.Features('3RectangleVertical',(x,y),maxWidth,maxHeight,-1,1))

# 3 Rectangle horizontal
def getAll3RectangleHorizontalFeatures(integralInmage,allfeature,initialWidth,initialHeight):
    maxHeight = max(initialHeight, 3)
    maxWidth = max(initialWidth, 1)
    for x in range((integralInmage.shape)[0] - maxHeight):
        for y in range((integralInmage.shape)[1] - maxWidth):
            allfeature.append(f.Features('3RectangleHorizontal',(x,y), maxWidth,maxHeight,1,1))
            allfeature.append(f.Features('3RectangleHorizontal',(x,y), maxWidth,maxHeight,-1,1))

# 4 Rectangle
def getAll4RectangleFeatures(integralInmage,allfeature,initialWidth,initialHeight):
    maxHeight = max(initialHeight, 2)
    maxWidth = max(initialWidth, 2)
    for x in range((integralInmage.shape)[0]-maxHeight):
        for y in range((integralInmage.shape)[1] - maxWidth):
            allfeature.append(f.Features('4Rectangle',(x,y),maxWidth,maxHeight,1,1))
            allfeature.append(f.Features('4Rectangle',(x,y),maxWidth,maxHeight,-1,1))

# get All features
def getAllFeatures(image,allfeature,initialWidth,initialHeight):
    for width in range(initialWidth, 10,1):
        for height in range(initialHeight, 10,2):
            getAll2RectangleHorizontalFeatures(image,allfeature,width,height)
    for width in range(initialWidth, 10,2):
        for height in range(initialHeight, 10,1):
            getAll2RectangleVerticalFeatures(image,allfeature,initialWidth,initialHeight)
    for width in range(initialWidth, 10,3):
        for height in range(initialHeight, 10,1):
            getAll3RectangleVerticalFeatures(image,allfeature,initialWidth,initialHeight)
    for width in range(initialWidth, 10,1):
        for height in range(initialHeight, 10,3):
            getAll3RectangleHorizontalFeatures(image,allfeature,initialWidth,initialHeight)
    for width in range(initialWidth, 10,2):
        for height in range(initialHeight, 10,2):
            getAll4RectangleFeatures(image,allfeature,initialWidth,initialHeight)

# calculate error for features
def getError(image, feature):
    return feature.getLable(image)

# according to the error, calculate lables for images
def getAllLable(tempFeature,image):
    allLables = []
    for i in range(len(tempFeature)):
        allLables.append(getError(image,tempFeature[i]))
    return allLables

# calculate the score of recognition
def examine(weakClassifiers, image):
    totalError = 0
    for feature in weakClassifiers:
        totalError = totalError + feature.getLable(image)
    if totalError >= 0:
        return 1
    else:
        return 0
  
# calculate the total number of correct recognition
def examineAll(weakClassifiers, integralImage):
    sum = 0
    for image in integralImage:
        sum = sum + examine(weakClassifiers,image)
    return sum

# create sub-windows
def sliding_window(image, windowSize):
    # slide a window across the image
    speed = 30
    for y in range(0, image.shape[0]-windowSize[1], round(image.shape[0]/speed)):
        for x in range(0, image.shape[1]-windowSize[0], round(image.shape[1]/speed)):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# examine the correctness for one image with power (to increase the window size)
def examineWithPower(weakClassifiers, image, powerW, powerH):
    totalError = 0
    counter = 1
    for feature in weakClassifiers:
        newLeftUp = (feature.leftUp[0] * powerH, feature.leftUp[1] * powerW)
        newFeature = f.Features(feature.type,newLeftUp,feature.width * powerW,feature.height * powerH,feature.polarity,2)
        totalError = totalError + newFeature.getLable(image)
        if counter == 1 and totalError <0:
            return 0
        counter +=1

    if totalError >= 0:
        return 1
    else:
        return 0

# extract one face using training features
def outPutFace(wearClassifiers,setHeight,setWidth, fileName):
    
    inputAddress = 'input'
    fileAddress = inputAddress +'/' + fileName
    i = 1
    # load the image and define the window width and height
    image = io.imread(fileAddress, as_grey=True)
    times = 5
    size = min(round(image.shape[1] / times), round(image.shape[0] / times))
    (winW, winH) = (size, size)
    windows = []
    
    while i<10:
        image = io.imread(fileAddress, as_grey=True)
        topLeftX = -1
        topLeftY = -1
        downRightX = 0
        downRightY = 0
        
        for (x, y, window) in sliding_window(image, windowSize=(winW, winH)):
            powerW = round(window.shape[1] / setWidth)
            powerH = round(window.shape[0] / setHeight)
        
            if examineWithPower(wearClassifiers, window, powerW, powerH) == 1:
                if topLeftX == -1 and topLeftY == -1:
                    topLeftX = x
                    topLeftY = y
                downRightX = max(downRightX,x+window.shape[1])
                downRightY = max(downRightY, y + window.shape[0])
            # clone = image.copy()
            # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            # cv2.imshow("Window", clone)
            # cv2.waitKey(1)
            
        tempImage = Image.open(fileAddress)
        newImage = tempImage.crop((topLeftX, topLeftY, downRightX, downRightY))
        newImage.save('output/' + fileName)
        i +=1
        fileAddress = 'output/' + fileName
    print('finished')

# AdaBoost algorithm
def AdaBoost(classifiersNum,weights,imageNumber,tempFeature,featureIndexs,allFeatures,lable):
    print('===== AdaBoosting =====')
    weakClassifiers = []
    for t in range(classifiersNum):
        weights = weights * 1.0 / sum(weights)
        minError = sys.maxsize
        minFeature = 0
        minFeatureIndex = 0

        for i in featureIndexs:
            totalError = 0
            for j in range(imageNumber):
                if allFeatures[j,i] != lable[j]:
                    totalError = totalError + weights[j]

            if totalError < minError:
                minError = totalError
                minFeature = tempFeature[i]
                minFeatureIndex = i

        # # update the feature weight
        minFeature.weight = 0.5 * np.log((1 - minError) / minError)

        weakClassifiers.append(minFeature)

        # update image weights
        for j in range(imageNumber):
            # print(weights[j])
            # print(np.sqrt((1-minError)/minError))
            if allFeatures[j,minFeatureIndex] != lable[j]:
                weights[j] = weights[j] * np.sqrt((1-minError)/minError)
            else:
                weights[j] = weights[j] * np.sqrt(minError / (1 - minError)),

        featureIndexs.remove(minFeatureIndex)
    print(str(len(weakClassifiers)) + ' weakClassifiers created')
    return weakClassifiers

# get all the integral images
def getAllIntegralImages(images):
    integral = []
    for i in images:
        integral.append(integralImage(i))
    return integral

# generate feature for an image
def generateFeature(imageNumber,integralFace,initialWidth,initialLength,integralImages):
    print('===== creating features =====')
    tempFeature = []
    getAllFeatures(integralFace[0], tempFeature, initialWidth, initialLength)
    featureSize = len(tempFeature)

    allFeatures = np.zeros((imageNumber,featureSize))

    for i in range(imageNumber):
        iiFace = integralImages[i]
        allLable = getAllLable(tempFeature,iiFace)
        allFeatures[i,:] = allLable

    featureIndexs = list(range(featureSize))
    print(str(len(featureIndexs)) + ' feature created')
    return (allFeatures,featureIndexs,tempFeature)

# assign weights
def getWeight(faceNum,otherNum):
    faceWeights = (1.0 * np.ones(faceNum))/ (2.0 * faceNum)
    otherWeights = (1.0 * np.ones(otherNum)) / (2.0 * otherNum)
    weights = np.hstack((faceWeights, otherWeights))
    weights = np.reshape(weights,(weights.shape[0],1))
    return weights

# examine the correctness for all test sets
def examineCorrectness(weakClassifiers):
    #load training faces and nonfaces
    testface = 'test/face'
    testnother = 'test/other'
    testfaceImages = loadImage(testface)
    testotherImages = loadImage(testnother)

    #convert face and nonface images to integral images
    print('===== creating integral images for test =====')
    testintegralFace = []
    testintegralOther = []
    for image in testfaceImages:
        testintegralFace.append(integralImage(image))

    for image in testotherImages:
        testintegralOther.append(integralImage(image))

    #calculate correctness
    correctFaces = examineAll(weakClassifiers,testintegralFace)
    correctOther = len(testintegralOther) - examineAll(weakClassifiers,testintegralOther)
    print('Result:\n      Faces: ' + str(correctFaces) + '/' + str(len(testintegralFace))
          + '  (' + str((float(correctFaces) / len(testintegralFace)) * 100) + '%)\n '+ 'non-Faces: '
          + str(correctOther) + '/' + str(len(testintegralOther)) + '  ('
          + str((float(correctOther) / len(testintegralOther)) * 100) + '%)')
    
# extract all faces using training features
def getFace(inputAddress,weakClassifiers):
    for filename in os.listdir(inputAddress):
        if 'pgm' in filename:
            tempImage = np.asarray(Image.open(inputAddress + '/' + filename))
            ii = integralImage(tempImage)
            if examine(weakClassifiers,ii) == 1:
                outPutFace(weakClassifiers, setHeight, setWidth, filename)

# extract all faces using CV2 (used as comparison)
def getFaceByCV2():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    for filename in os.listdir('input'):
        if 'pgm' in filename:
            img = cv2.imread('input/' + filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                tempImage = Image.open('input/' + filename)
                newImage = tempImage.crop((x, y, x + w, y + h))
                newImage.save('output2/' + filename)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# main function
if __name__ == "__main__":
    classifiersNum = 2
    initialWidth = 8
    initialLength = 8
    print('classifiers: ' + str(classifiersNum) + '; initial width/height: ' + str(initialWidth))

    # load training faces and nonfaces
    trainface = 'training/face'
    trainother = 'training/other'

    faceImages = loadImage(trainface)
    otherImages = loadImage(trainother)

    # convert face and nonface images to integral images
    print('===== creating integral images =====')
    integralFace = getAllIntegralImages(faceImages)
    integralOther = getAllIntegralImages(otherImages)

    # assign weight/label to images
    faceNum = len(integralFace)
    otherNum = len(integralOther)

    # get label
    lable =  np.hstack((np.ones(faceNum), np.ones(otherNum) * -1))

    integralImages = integralFace+integralOther
    # get weight
    weights = getWeight(faceNum,otherNum)

    # print('===== creating features =====')
    imageNumber = faceNum + otherNum
    (allFeatures, featureIndexs,tempFeature) = generateFeature(imageNumber,integralFace,initialWidth,initialLength,integralImages)

    # AdaBoost to get best performance feature
    weakClassifiers = AdaBoost(classifiersNum,weights,imageNumber,tempFeature,featureIndexs,allFeatures,lable)

    # examine Correctness
    examineCorrectness(weakClassifiers)

    # extract faces
    setWidth = integralFace[0].shape[1]
    setHeight = integralFace[0].shape[0]
    inputAddress = 'input'

    getFace(inputAddress, weakClassifiers)
    getFaceByCV2()

    
    

            
    
