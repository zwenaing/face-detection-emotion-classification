import numpy as np

class Features():
    def __init__(self, type, leftUp, width, height, polarity,stage):
        self.type = type
        self.leftUp = leftUp
        self.width = width
        self.height = height
        self.polarity = polarity
        self.weight = 1
        self.stage = stage
    
    def getArea(self,leftTop, rightdown, integral):
        leftdown = (int(rightdown[0]), int(leftTop[1]))
        rightup = (int(leftTop[0]), int(rightdown[1]))
        rightdown = (int(rightdown[0]), int(rightdown[1]))
        leftup = (int(leftTop[0]), int(leftTop[1]))
        total = integral[rightdown] - integral[rightup] - integral[leftdown] + integral[leftup]
        return total
    
    def getArea2(self,leftTop, rightdown, integral):
        leftdown = (int(rightdown[0]), int(leftTop[1]))
        rightup = (int(leftTop[0]), int(rightdown[1]))
        rightdown = (int(rightdown[0]), int(rightdown[1]))
        leftup = (int(leftTop[0]), int(leftTop[1]))
        four = np.int(str(integral[rightdown]))
        two = np.int(str(integral[rightup]))
        three = np.int(str(integral[leftdown]))
        one = np.int(str(integral[leftup]))
        total = four - two - three + one
        return total
    
    def getLable(self,image):
        x = self.leftUp[0]
        y = self.leftUp[1]
        if self.type == '2RectangleHorizontal':
            rightdown = (x+self.height/2,y+self.width)
            blackLeftUp = (x + self.height / 2, y)
            blackRightDown = (x+self.height,y+self.width)
            if self.stage == 1:
                value = self.getArea(self.leftUp,rightdown,image) - self.getArea(blackLeftUp,blackRightDown,image)
            else:
                value = self.getArea2(self.leftUp,rightdown,image) - self.getArea2(blackLeftUp,blackRightDown,image)

        elif self.type == '2RectangleVertical':
            rightdown = (x+self.height,y+self.width/2)
            blackLeftUp = (x,y+self.width/2)
            blackRightDown = (x+self.height,y+self.width)
            if self.stage == 1:
                value = self.getArea(self.leftUp, rightdown, image) - self.getArea(blackLeftUp, blackRightDown, image)
            else:
                value = self.getArea2(self.leftUp, rightdown, image) - self.getArea2(blackLeftUp, blackRightDown, image)

        elif self.type == '3RectangleVertical':
            rightdown = (x+self.height,y+self.width/3)
            blackLeftUp = (x,y+self.width/3)
            blackRightDown = (x+self.height,y+self.width*2/3)
            white2LeftUp = (x,y+self.width*2/3)
            white2RightDown = (x + self.height, y + self.width)
            if self.stage == 1:
                value = self.getArea(self.leftUp, rightdown, image) + self.getArea(white2LeftUp, white2RightDown,
                                                                                   image) - \
                        self.getArea(blackLeftUp, blackRightDown, image)
            else:
                value = self.getArea2(self.leftUp, rightdown, image) + self.getArea2(white2LeftUp, white2RightDown,
                                                                                   image) - \
                        self.getArea2(blackLeftUp, blackRightDown, image)

        elif self.type == '3RectangleHorizontal':
            rightdown =(x+self.height/3,y+self.width)
            blackLeftUp = (x+self.height/3,y)
            blackRightDown = (x+self.height*2/3,y+self.width)
            white2LeftUp = (x+self.height*2/3,y)
            white2RightDown = (x+self.height,y+self.width)
            if self.stage == 1:
                value = self.getArea(self.leftUp, rightdown, image) + self.getArea(white2LeftUp, white2RightDown,
                                                                                   image) - \
                        self.getArea(blackLeftUp, blackRightDown, image)
            else:
                value = self.getArea2(self.leftUp, rightdown, image) + self.getArea2(white2LeftUp, white2RightDown,
                                                                                   image) - \
                        self.getArea2(blackLeftUp, blackRightDown, image)

        else:
            rightdown =(x+self.height/2,y+self.width/2)
            black1LeftUp = (x,y+self.width/2)
            black1RightDown = (x+self.height/2,y+self.width)
            white2LeftUp = (x+self.height/2,y)
            white2RightDown = (x+self.height,y+self.width/2)
            black2LeftUp = (x+self.height/2,y+self.width/2)
            black2RightDown =  (x+self.height,y+self.width)
            if self.stage == 1:
                value = self.getArea(self.leftUp, rightdown, image) + \
                        self.getArea(white2LeftUp, white2RightDown, image) - \
                        self.getArea(black1LeftUp, black1RightDown, image) - \
                        self.getArea(black2LeftUp, black2RightDown, image)
            else:
                value = self.getArea2(self.leftUp, rightdown, image) + \
                        self.getArea2(white2LeftUp, white2RightDown, image) - \
                        self.getArea2(black1LeftUp, black1RightDown, image) - \
                        self.getArea2(black2LeftUp, black2RightDown, image)

        
        if value*self.polarity < 0:
            return 1 * self.weight
        else:
            return -1 * self.weight
            

#
# # 2 Rectangle vertical
#
# def get2RectangleVerticalFeatures(image,whiteLeftUp,whiteRightDown,blackLeftUp,blackRightDown):
#     return  getArea(whiteLeftUp,whiteRightDown,image) - getArea(blackLeftUp,blackRightDown,image)
#
# def getAll2RectangleVerticalFeatures(integralInmage,allfeature,initialWidth,initialHeight):
#     maxHeight = max(initialHeight, 1)
#     maxWidth = max(initialWidth, 2)
#     for x in range((integralInmage.shape)[0] - maxHeight):
#         for y in range((integralInmage.shape)[1] - maxWidth):
#             allfeature.append(get2RectangleVerticalFeatures(integralInmage,(x,y),(x+maxHeight,y+maxWidth/2),
#                                                             (x,y+maxWidth/2),(x+maxHeight,y+maxWidth)))
#             allfeature.append(-1 * get2RectangleVerticalFeatures(integralInmage, (x, y), (x + maxHeight, y + maxWidth / 2),
#                                                         (x, y + maxWidth / 2), (x + maxHeight, y + maxWidth)))

# # 3 Rectangle vertical
# def get3RectangleVerticalFeatures(image,white1LeftUp,white1RightDown,blackLeftUp,blackRightDown,white2LeftUp,white2RightDown):
#     return getArea(white1LeftUp,white1RightDown,image) + getArea(white2LeftUp,white2RightDown,image) - getArea(blackLeftUp,blackRightDown,image)
#
# def getAll3RectangleVerticalFeatures(integralInmage,allfeature,initialWidth,initialHeight):
#     maxHeight = max(initialHeight, 1)
#     maxWidth = max(initialWidth, 3)
#     for x in range((integralInmage.shape)[0] - maxHeight):
#         for y in range((integralInmage.shape)[1] - maxWidth):
#             allfeature.append(get3RectangleVerticalFeatures(integralInmage,(x,y),(x+maxHeight,y+maxWidth/3),
#                                                             (x,y+maxWidth/3),(x+maxHeight,y+maxWidth*2/3),
#                                                             (x,y+maxWidth*2/3),(x+maxHeight,y+maxWidth)))
#             allfeature.append(-1 * get3RectangleVerticalFeatures(integralInmage, (x, y), (x + maxHeight, y + maxWidth / 3),
#                                                         (x, y + maxWidth / 3), (x + maxHeight, y + maxWidth * 2 / 3),
#                                                         (x, y + maxWidth * 2 / 3), (x + maxHeight, y + maxWidth)))

# 3 Rectangle horizontal
# def get3RectangleHorizontalFeatures(image,white1LeftUp,white1RightDown,blackLeftUp,blackRightDown,white2LeftUp,white2RightDown):
#     return getArea(white1LeftUp,white1RightDown,image) + getArea(white2LeftUp,white2RightDown,image) - getArea(blackLeftUp,blackRightDown,image)
#
# def getAll3RectangleHorizontalFeatures(integralInmage,allfeature,initialWidth,initialHeight):
#     maxHeight = max(initialHeight, 3)
#     maxWidth = max(initialWidth, 1)
#     for x in range((integralInmage.shape)[0] - maxHeight):
#         for y in range((integralInmage.shape)[1] - maxWidth):
#             allfeature.append(get3RectangleHorizontalFeatures(integralInmage, (x,y), (x+maxHeight/3,y+maxWidth),
#                                                               (x+maxHeight/3,y),(x+maxHeight*2/3,y+maxWidth),
#                                                               (x+maxHeight*2/3,y),(x+maxHeight,y+maxWidth)))
#             allfeature.append(-1 * get3RectangleHorizontalFeatures(integralInmage, (x, y), (x + maxHeight / 3, y + maxWidth),
#                                                           (x + maxHeight / 3, y), (x + maxHeight * 2 / 3, y + maxWidth),
#                                                           (x + maxHeight * 2 / 3, y), (x + maxHeight, y + maxWidth)))


# 4 Rectangle
# def get4RectangleFeatures(image,white1LeftUp,white1RightDown,black1LeftUp,black1RightDown,white2LeftUp,white2RightDown,black2LeftUp,black2RightDown,):
#     return getArea(white1LeftUp,white1RightDown,image) + getArea(white2LeftUp,white2RightDown,image) - \
#            getArea(black1LeftUp,black1RightDown,image) - getArea(black2LeftUp,black2RightDown,image)
#
# def getAll4RectangleFeatures(integralInmage,allfeature,initialWidth,initialHeight):
#     maxHeight = max(initialHeight, 2)
#     maxWidth = max(initialWidth, 2)
#     for x in range((integralInmage.shape)[0]-maxHeight):
#         for y in range((integralInmage.shape)[1] - maxWidth):
#             allfeature.append(get4RectangleFeatures(integralInmage,(x,y),(x+maxHeight/2,y+maxWidth/2),
#                                                     (x,y+maxWidth/2),(x+maxHeight/2,y+maxWidth),
#                                                     (x+maxHeight/2,y),(x+maxHeight,y+maxWidth/2),
#                                                     (x+maxHeight/2,y+maxWidth/2),(x+maxHeight,y+maxWidth)))
#             allfeature.append(-1 * get4RectangleFeatures(integralInmage, (x, y), (x + maxHeight / 2, y + maxWidth / 2), (x, y + maxWidth / 2),
#                                   (x + maxHeight / 2, y + maxWidth), (x + maxHeight / 2, y),
#                                   (x + maxHeight, y + maxWidth / 2), (x + maxHeight / 2, y + maxWidth / 2),
#                                   (x + maxHeight, y + maxWidth)))
#
#
# # get All features
# def getAllFeatures(image,allfeature,initialWidth,initialHeight):
#     for width in range(initialWidth, 10,1):
#         for height in range(initialHeight, 10,2):
#             getAll2RectangleHorizontalFeatures(image,allfeature,width,height)
#     for width in range(initialWidth, 10,2):
#         for height in range(initialHeight, 10,1):
#             getAll2RectangleVerticalFeatures(image,allfeature,initialWidth,initialHeight)
#     for width in range(initialWidth, 10,3):
#         for height in range(initialHeight, 10,1):
#             getAll3RectangleVerticalFeatures(image,allfeature,initialWidth,initialHeight)
#     for width in range(initialWidth, 10,1):
#         for height in range(initialHeight, 10,3):
#             getAll3RectangleHorizontalFeatures(image,allfeature,initialWidth,initialHeight)
#     for width in range(initialWidth, 10,2):
#         for height in range(initialHeight, 10,2):
#             getAll4RectangleFeatures(image,allfeature,initialWidth,initialHeight)
#

