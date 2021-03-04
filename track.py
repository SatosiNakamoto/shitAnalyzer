from matplotlib import pyplot as plt
import os
import time
import cv2 as cv
import math
import numpy as np


upperBorder = 5700  # 5700
bottomBorder = 1000
signTrue = lambda a: (a > 0) - (a < 0)


def sign(a):
    if a > 0:
        return 1
    elif a < 0:
        return -1
    else:
        return 0


class MapFrameType():
    def __init__(self, date, pathImg=None, pathArr=None):
        self.date = date
        self.pathImg = pathImg
        self.pathArr = pathArr
        self.frame = None
        self.map = None
        for i in os.listdir(self.pathImg):
            if i.split(".")[0] == self.date:
                self.frame = Frame(cv.imread(self.pathImg + i))
        for i in os.listdir(self.pathArr):
            if i.split(".")[0] == self.date:
                self.map = Map(np.loadtxt(self.pathArr + i, skiprows=0))

    def prepareFeatures(self):
        self.map.setFeaturePoints(self.frame.getFeaturesPoints())

    def gradientmap(self):
        gr = np.gradient(self.map.map)
        plt.imshow(gr)
        plt.show()

    def imageSegmentation(self):
        spatialRadius = 35;
        colorRadius = 60;
        pyramidLevels = 3;
        imageSegment = None
        imageSegment = cv.pyrMeanShiftFiltering(self.frame.img, imageSegment, spatialRadius, colorRadius,
                                                pyramidLevels);
        cv.imshow("segmented",imageSegment)
        cv.waitKey(0)
        return imageSegment


class Map():
    def __init__(self, map):
        self.map = map
        self.upperBorder = 5700  # 5700
        self.bottomBorder = 1000
        self.processMap()

    def processMap(self):
        self.map[self.map <= self.bottomBorder] = 0
        self.map[self.map >= self.upperBorder] = 0

    def setFeaturePoints(self, points):
        self.map[300, 300] = np.nan
        print(self.map[300, 300])
        print("######################")
        for p in points:
            if p[0] >= 480 or p[1] >= 640:
                continue
            #print(str(int(p[0])) + " " + str(int(p[1])))
            self.map[int(p[0]), int(p[1])] = np.nan
        print("######################")

    def disaideWhichHalfIsCloserToCar(self):
        halfOfColumns = int(self.map.shape[1] / 2)
        window = self.map[0:self.map.shape[1], 0:halfOfColumns]
        leftHalf = window.sum()
        rightHalf = self.map.sum() - leftHalf
        #print("right half - " + str(rightHalf))
        #print("left half  - " + str(leftHalf))

        if leftHalf > rightHalf:
            return 0
        else:
            return 1

    def showMap(self):
        #plt.imshow(self.map)
        return


class Frame():
    def __init__(self, img):
        self.img = img
        self.orb = cv.ORB_create()

    def getFeaturesPoints(self):
        if self.img is not None:
            gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
            dst = cv.cornerHarris(gray, 2, 3, 0.099)
            feats = cv.goodFeaturesToTrack(np.mean(self.img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01,
                                           minDistance=3)
            kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
            kps, des = self.orb.compute(self.img, kps)
            return (kps, des)

    def drawFeaturesPoints(self, points=None):
        if points == None:
            print("None")
            points = self.getFeaturesPoints()[0]
        for p in points:
            if p.pt[1] > 480 or p.pt[0] > 640:
                continue
            cv.circle(self.img, (int(p.pt[0]), int(p.pt[1])), 5, (0), 2)

    def showFrame(self):
        cv.imshow("frame", self.img)


class mapProcessor():
    def __init__(self, p):
        self.path = p
        self.upperBorder = 5700  # 5700
        self.bottomBorder = 1000
        self.processedMaps = []

    def getMaps(self, n):
        imgs = []
        c = 0
        for i in os.listdir(self.path):
            if c > n:
                break
            c = c + 1

            img = np.loadtxt(self.path + i, skiprows=0)

            imgs.append(Map(img))
        return imgs

    def processNMaps(self, n):
        for map in self.getMaps(n):
            map.map[map.map <= self.bottomBorder] = 0
            map.map[map.map >= self.upperBorder] = 0
            self.processedMaps.append(map)

    def splitCarsByItsMaps(self):
        return None


class Matcher():
    def __init__(self):
        self.bf = cv.BFMatcher(cv.NORM_HAMMING)
        self.last = None
        self.delta = 0
        self.perfectMatches = []
        self.kp1 = []
        self.kp2 = []
        self.img1 = None
        self.img2 = None
        self.map1 = None
        self.map2 = None
        self.frame1 = None
        self.frame2 = None
        self.mapShif = None
        self.sideWhereMoveingFrom = -1
        self.contourCoor = None

    def drawMatches(self):
        rows1 = self.img1.shape[0]
        cols1 = self.img1.shape[1]
        rows2 = self.img2.shape[0]
        cols2 = self.img2.shape[1]

        out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
        out[:rows1, :cols1] = np.dstack([self.img1, self.img1, self.img1])
        out[:rows2, cols1:] = np.dstack([self.img2, self.img2, self.img2])
        #print("-----------------")
        k = 0
        dx = []
        dy = []
        if self.contourCoor == None:
            return
        for mat in self.perfectMatches:
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            (x1, y1) = self.kp1[img1_idx].pt
            (x2, y2) = self.kp2[img2_idx].pt

            if self.contourCoor[0] < x1 < self.contourCoor[2]:
                if self.contourCoor[0] < x2 < self.contourCoor[2]:
                    if self.contourCoor[1] < y1 < self.contourCoor[3]:
                        if self.contourCoor[1] < y2 < self.contourCoor[3]:
                            dx.append(x1 - x2)
                            dy.append(y1 - y2)
                            k += 1
                            cv.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
                            cv.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)
                            cv.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)
                            #print(str(x1) + " " + str(y1) + "                 " + str(x2) + " " + str(y2))
        #print(str(dx) + " " + str(dy))
        #print(k)
        #plt.show()
        medianX.append(np.median(dx))
        medianY.append(np.median(dy))

        #print('x', stats.mode(dx), np.median(dx))
        #print('y', stats.mode(dy), np.median(dy))
        #print("-----------------")
        #cv.line(out, (640 + int(self.delta), 0), (640 + int(self.delta), 480), (0, 255, 0), thickness=2)
        #cv.imshow('Matched Features', out)


        #plt.show()


        return out

    def compareTwoFrame(self, mapFrame1, mapFrame2):
        #print(mapFrame2, mapFrame1)
        features1 = mapFrame1.frame.getFeaturesPoints()
        features2 = mapFrame2.frame.getFeaturesPoints()
        self.img1 = cv.cvtColor(mapFrame1.frame.img, cv.COLOR_BGR2GRAY)
        self.img2 = cv.cvtColor(mapFrame2.frame.img, cv.COLOR_BGR2GRAY)
        self.map1 = mapFrame1.map
        self.frame1 = mapFrame1.frame
        self.map2 = mapFrame2.map
        self.frame2 = mapFrame2.frame
        matches = self.bf.knnMatch(features1[1], features2[1], k=2)
        retdescs = []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                retdescs.append(m)

        self.kp1 = features1[0]
        self.kp2 = features2[0]
        sumSpeed = 0
        countOfChangedPoints = 0
        directionSum = 0
        for mat in retdescs:
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            (x1, y1) = self.kp1[img1_idx].pt
            (x2, y2) = self.kp2[img2_idx].pt

            if abs(x2 - x1) < 5 and abs(y2 - y1) < 5:
                continue

            self.perfectMatches.append(mat)
            pathVectorLen = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
            sumSpeed = sumSpeed + pathVectorLen
            countOfChangedPoints = countOfChangedPoints + 1
            directionSum = directionSum + sign(x2 - x1)
        if countOfChangedPoints == 0:
            self.delta = 0
        else:
            self.delta = int(sumSpeed / countOfChangedPoints) * sign(directionSum)
        #print("delta is" + str(self.delta))
        return self.perfectMatches

    def findCountoursOftruck(self):
        diff = cv.absdiff(self.img1, self.img2)
        blur = cv.GaussianBlur(diff, (5, 5), 0)
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        сontours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #print("cont len is" + str(len(сontours)))
        good_cont = []
        for contour in сontours:
            (x, y, w, h) = cv.boundingRect(contour)
            #print(cv.contourArea(contour))
            if cv.contourArea(contour) < 20000 or cv.contourArea(contour) > 100000:
                continue
            good_cont.append(contour)
            cv.rectangle(self.img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.contourCoor = [x, y, x + w, y + h]
            #print(self.contourCoor)
        sq = 0
        m = []
        for contour in сontours:
            if cv.arcLength(contour, True) > sq:
                sq = cv.arcLength(contour, True)
                m = contour
        cv.drawContours(self.img2, m, -1, (0, 255, 255), 2)

    def findFirstMapMask(self):
        self.contourCoor = None

    def calcMapShift(self):
        if self.delta > 0:
            self.sideWhereMoveingFrom = 0
        elif self.delta < 0:
            self.sideWhereMoveingFrom = 1
        else:
            self.sideWhereMoveingFrom = -1

        #print("side is" + str(self.sideWhereMoveingFrom))
        #print("delta is " + str(self.delta))
        if self.sideWhereMoveingFrom == 1:
            self.mapShif = Map(
                self.map2.map[0:self.map2.map.shape[0], self.map2.map.shape[1] + self.delta:self.map2.map.shape[1]])
        elif self.sideWhereMoveingFrom == 0:
            self.mapShif = Map(self.map1.map[0:self.map1.map.shape[0], 0:self.delta])


def paintDepthMap(biasX, biasY, listDepthMapPath):
    #print(listDepthMapPath)
    endDepthMap = np.zeros((1500, 3000))
    endMask = np.zeros((1500, 3000))
    x = 0
    y = 0
    if biasX > 0:
        x = 1500
        y = 0
    biasX = int(-1 * biasX)
    biasY = int(-1 * biasY)
    for depthMapPath in listDepthMapPath:
        depthMap = np.loadtxt(depthMapPath)
        for i in range(depthMap.shape[0]):
            for j in range(depthMap.shape[1]):
                if endDepthMap[i + y][j + x] == 0:
                    endDepthMap[i + y][j + x] = depthMap[i][j]
                elif endDepthMap[i + y][j + x] != 0 and depthMap[i][j] != 0:
                    endDepthMap[i + y][j + x] = depthMap[i][j]
        mask = findCountors(depthMap.copy())
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if endMask[i + y][j + x] == 0:
                    endMask[i + y][j + x] = mask[i][j][0]
                elif endMask[i + y][j + x] != 0 and mask[i][j][0] != 0:
                    endMask[i + y][j + x] = mask[i][j][0]
        x += biasX
        y += biasY
    for i in range(endDepthMap.shape[0]):
        for j in range(endDepthMap.shape[1]):
            if endMask[i][j] == 0:
                endDepthMap[i][j] = 0
    endDepthMap[endDepthMap > 7000] = 0
    return endDepthMap
    #plt.subplot(1, 2, 1)
    #plt.imshow(endDepthMap, interpolation='nearest')
    #plt.subplot(1, 2, 2)
    #plt.imshow(endMask, interpolation='nearest')
    #plt.show()


def findCountors(frame):
    x = np.mean(frame)
    frame[frame <= 1] = 0.0
    frame[frame > x] = 0.0
    frame[frame > 0] = 255.0
    cv.imwrite("test.jpg", frame)
    x = cv.imread("test.jpg")
    imgray = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
    imgray = cv.GaussianBlur(imgray, (3, 3), 0)
    contours, hierarchy = cv.findContours(imgray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sq = 0
    m = []
    for contour in contours:
        if cv.contourArea(contour) > sq:
            sq = cv.contourArea(contour)
            m = contour
    x[x < 256] = 0
    cv.fillConvexPoly(x, cv.convexHull(m), color=(255, 255, 255))
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if x[i][j][0] == 0:
                frame[i][j] = 0
    return x




array = "D:/svt/array12/"
images = "D:/svt/image12/"
tmLast = 0
lastImages = 0
medianX = []
medianY = []
k = 0
count = 1
listDepthMap = []
for i in os.listdir(images):
    tmNext = time.mktime(time.strptime(i[:19], '%d-%m-%Y-%H-%M-%S'))
    if (tmNext < tmLast + 5):
        if k != 0:
            mf1 = MapFrameType(i[:-4], images, array)
            mf2 = MapFrameType(lastImages[:-4], images, array)
            matcher = Matcher()
            matchesList = matcher.compareTwoFrame(mf1, mf2)
            cv.waitKey(0)
            matcher.findCountoursOftruck()
            matcher.drawMatches()
            matcher.calcMapShift()
            #matcher.mapShif.showMap()
            matcher.contourCoor
            lastImages = i
        else:
            lastImages = i
            k = 1
    else:
        x = np.median(medianX[1:7])
        y = np.median(medianY[1:7])
        #print(medianX, x)  # х
        #print(medianY, y)  # y
        if len(listDepthMap) != 0:
            picture = paintDepthMap(x, y, listDepthMap)
            picture = picture / np.max(picture) * 255
            cv.imwrite(i, picture)
        medianX = []
        medianY = []
        print("new images")
        #plt.subplot(6, 8, count)
        '''if count == 48:
            plt.show()
            count = 1
        else:
            count += 1'''
        listDepthMap = []
        k = 0
    listDepthMap.append(array + i[:-4] + ".txt")
    tmLast = tmNext