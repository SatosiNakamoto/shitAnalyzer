from matplotlib import pyplot as plt
import os
import time
import cv2 as cv
import math
import numpy as np
from scipy.ndimage.interpolation import shift

def sign(a):
    if a > 0:
        return 1
    elif a < 0:
        return -1
    else:
        return 0

class TimeMeassureUtil():
    def __init__(self, mes):
        self.message = mes
        self.startMeassureTime = time.process_time()
    
    def meassureTime(self):
        print(self.message, time.process_time() - self.startMeassureTime)
        self.startMeassureTime = 0


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


class Map():
    def __init__(self, map):
        self.map = map
        self.upperBorder = 5700
        self.bottomBorder = 1000
        self.processMap()

    def processMap(self):
        self.map[self.map <= self.bottomBorder] = 0
        self.map[self.map >= self.upperBorder] = 0

    def setFeaturePoints(self, points):
        self.map[300, 300] = np.nan
        for p in points:
            if p[0] >= 480 or p[1] >= 640:
                continue
            self.map[int(p[0]), int(p[1])] = np.nan

    def showMap(self):
        plt.imshow(self.map)
        plt.show()


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

    def showFrame(self):
        cv.imshow("frame", self.img)


class Matcher():
    listForMedianX = []  # static variable - для всех матчеров
    listForMedianY = []

    def __init__(self):
        self.bf = cv.BFMatcher(cv.NORM_HAMMING)
        self.dxList = []
        self.dyList = []
        self.perfectMatches = []
        self.kp1 = []
        self.kp2 = []
        self.img1 = None
        self.img2 = None
        self.map1 = None
        self.map2 = None
        self.frame1 = None
        self.frame2 = None
        self.contourCoor = None

    def compareTwoFrame(self, mapFrame1, mapFrame2):
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
        return self.perfectMatches

    def findCountoursOftruck(self):
        diff = cv.absdiff(self.img1, self.img2)
        blur = cv.GaussianBlur(diff, (5, 5), 0)
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        сontours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        good_cont = []
        for contour in сontours:
            (x, y, w, h) = cv.boundingRect(contour)
            if cv.contourArea(contour) < 20000: # or cv.contourArea(contour) > 200000:
                continue
            good_cont.append(contour)
            # добавить фильтрацию контуров
            self.contourCoor = [x, y, x + w, y + h]

    def findFirstMapMask(self):
        self.contourCoor = None

    def calcDeltaVectorsOfFeaturesInsideContour(self):
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
                            self.dxList.append(x2 - x1)
                            self.dyList.append(y2 - y1)
        if len(self.dxList) != 0 and len(self.dyList) != 0:
            Matcher.listForMedianX.append(np.median(self.dxList))
            Matcher.listForMedianY.append(np.median(self.dyList))
        else:
            Matcher.listForMedianX.append(Matcher.listForMedianX[-1])
            Matcher.listForMedianY.append(Matcher.listForMedianY[-1])

    def getTotalMotionVector():
        Matcher.motionVector = [0, 0]
        for i in Matcher.listForMedianX:
            Matcher.motionVector[0] += i
        for i in Matcher.listForMedianY:
            Matcher.motionVector[1] += i


class Truck():
    def __init__(self, mapsList, biasX, biasY, endMapSizeX=700, endMapSizeY=1800):
        self.listDepthMapPath = mapsList
        self.endMapSizeX = endMapSizeX
        self.endMapSizeY = endMapSizeY
        self.endTruckDepthMap = np.zeros((endMapSizeX, endMapSizeY))
        self.endTruckMask = np.zeros((endMapSizeX, endMapSizeY))
        self.biasX = int(biasX)
        self.biasY = int(biasY)
        self.width = -1
        self.treshFilterSmoke = 4200
        self.treshFilterGround = 6800
        self.acceleratedMovementX = 0
        self.acceleratedMovementY = 0
        self.volume = 0

    def showTruckMap(self):
        plt.imshow(self.endTruckDepthMap)
        plt.show()

    def findCountors(self, frame):
        x = np.mean(frame)
        frame[frame <= 1] = 0.0
        frame[frame > x] = 0.0
        frame[frame > 0] = 255.0
        cv.imwrite("test.jpg", frame)
        x = cv.imread("test.jpg")
        imgray = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
        imgray = cv.GaussianBlur(imgray, (3, 3), 0)
        contours, hierarchy = cv.findContours(imgray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        segmented = max(contours, key=cv.contourArea)
        x[x < 256] = 0
        cv.drawContours(x, segmented, -1, (255, 255, 255), 1)
        imgray = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
        contours2, hierarchy2 = cv.findContours(imgray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.fillPoly(x, contours2, (255, 255, 255))
        return x

    def getTruckWidth(self):
        self.width = np.sum(self.endTruckDepthMap) / (self.endTruckDepthMap.shape[0] * self.endTruckDepthMap.shape[1])

    def getEquallyAcceleratedMovement(self, medianX, medianY):
        tm = TimeMeassureUtil("work of getEquallyAcceleratedMovement")
        if len(medianX) > 4:
            self.acceleratedMovementX = (np.mean(medianX[1:4]) - np.mean(medianX[-4:-1])) / (len(medianX) - 4)
            self.acceleratedMovementY = (np.mean(medianY[1:4]) - np.mean(medianY[-4:-1])) / (len(medianY) - 4)
        x = 0
        y = 0
        if self.biasX < 0:
            x = x + self.endMapSizeX + 120
        count = 0
        first = True
        for depthMapPath in self.listDepthMapPath:
            depthMap = np.loadtxt(depthMapPath)
            depthMap[depthMap < self.treshFilterSmoke] = 0
            mask = self.findCountors(depthMap.copy())
            firstDim = range(depthMap.shape[0])
            secondDim = range(depthMap.shape[1])
            
            bigMapForDepthMap = np.zeros((self.endTruckDepthMap.shape[0], self.endTruckDepthMap.shape[1]))
            bigMapForDepthMap[0 : depthMap.shape[0], 0 : depthMap.shape[1]] = depthMap
            
            #plt.imshow(bigMapForDepthMap)
            #plt.show()
            '''
            for i in firstDim:
                for j in secondDim:
                    frameShiftX = int(j + x + round(self.acceleratedMovementX * count))
                    frameShiftY = int(i + y + round(self.acceleratedMovementY * count))
                    if frameShiftX >= self.endMapSizeY or frameShiftY >= self.endMapSizeX:
                        continue
                    if self.endTruckDepthMap[frameShiftY][frameShiftX] == 0:
                        self.endTruckDepthMap[frameShiftY][frameShiftX] = depthMap[i][j]
                    elif self.endTruckDepthMap[frameShiftY][frameShiftX] != 0 and depthMap[i][j] != 0:
                        self.endTruckDepthMap[frameShiftY][frameShiftX] = (self.endTruckDepthMap[frameShiftY][frameShiftX] + depthMap[i][j]) / 2
                    #if mask[i][j][0] != 0: # тут putmask добавить надо
                    #    self.endTruckMask[frameShiftY][frameShiftX] = mask[i][j][0]
            '''
            
            frameShiftX = int(x + round(self.acceleratedMovementX * count))
            frameShiftY = int(y + round(self.acceleratedMovementY * count))
            
            bigMapForDepthMap = np.roll(bigMapForDepthMap, frameShiftX, axis = 1)
            bigMapForDepthMap = np.roll(bigMapForDepthMap, frameShiftY, axis = 0)
            
            if first:
                self.endTruckDepthMap[0 : bigMapForDepthMap.shape[0], 0 : bigMapForDepthMap.shape[1]] = bigMapForDepthMap
                first = False
            else:        
                m = np.where(self.endTruckDepthMap == 0, 0, np.nan)# = bigMapForDepthMap
                m[0 : bigMapForDepthMap.shape[0], 0 : bigMapForDepthMap.shape[1]] = bigMapForDepthMap #+
                mm = np.where(self.endTruckDepthMap != 0, 0, np.nan)
                mm += bigMapForDepthMap
                mm /= 2
                m = np.where(np.isnan(m) , mm, m)    
                tempEndTruckDepthMap = self.endTruckDepthMap.copy()
                self.endTruckDepthMap += m 
                self.endTruckDepthMap[tempEndTruckDepthMap < self.endTruckDepthMap]/=2
			
            count += 1
            x += self.biasX
            y += self.biasY
		
        plt.imshow(self.endTruckDepthMap)
        plt.show()
        np.putmask(self.endTruckDepthMap, self.endTruckMask == 0.0, 0)
        self.endTruckDepthMap[self.endTruckDepthMap > self.treshFilterGround] = 0
        tm.meassureTime()
        return self.endTruckDepthMap

    def getVolume(self):
        ground = np.max(self.endTruckDepthMap)
        for i in self.endTruckDepthMap:
            for j in i:
                if j != 0.0:
                    self.volume += ground - j
        return self.volume


class Application():
    def __init__(self, arrayPath, imagePath):
        self.arrayPath = arrayPath
        self.imagePath = imagePath
        self.writeMode = False
        self.listDepthMap = []

    def run(self):
        tmLast = 0
        lastImages = 0
        count = 1
        for i in os.listdir(self.imagePath):
            tmNext = time.mktime(time.strptime(i[:19], '%d-%m-%Y-%H-%M-%S'))
            if (tmNext < tmLast + 5):
                if tmLast != 0:
                    mf1 = MapFrameType(i[:-4], self.imagePath, self.arrayPath)
                    mf2 = MapFrameType(lastImages[:-4], self.imagePath, self.arrayPath)
                    matcher = Matcher()
                    matchesList = matcher.compareTwoFrame(mf1, mf2)
                    matcher.findCountoursOftruck()
                    matcher.calcDeltaVectorsOfFeaturesInsideContour()
            else:
                x = np.median(Matcher.listForMedianX[1:-1])
                y = np.median(Matcher.listForMedianY[1:-1])
                if len(self.listDepthMap) != 0:
                    Matcher.getTotalMotionVector()
                    print("---------------")
                    print(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime(tmLast)))
                    tr = Truck(self.listDepthMap, x, y)
                    picture = tr.getEquallyAcceleratedMovement(Matcher.listForMedianX, Matcher.listForMedianY)
                    tr.showTruckMap()
                    print(tr.getVolume())
                    plt.subplot(2, 3, count)
                    plt.imshow(picture)
                    if count == 6:
                        plt.show()
                        count = 1
                    else:
                        count += 1
                Matcher.listForMedianX = []
                Matcher.listForMedianY = []
                self.listDepthMap = []
            lastImages = i
            self.listDepthMap.append(self.arrayPath + i[:-4] + ".txt")
            tmLast = tmNext
        x = np.median(Matcher.listForMedianX[1:-1])
        y = np.median(Matcher.listForMedianY[1:-1])
        if len(self.listDepthMap) != 0:
            Matcher.getTotalMotionVector()
            print("---------------")
            print(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime(tmLast)))
            tr = Truck(self.listDepthMap, x, y)
            picture = tr.getEquallyAcceleratedMovement(Matcher.listForMedianX, Matcher.listForMedianY)
            print(tr.getVolume())
            plt.subplot(2, 3, count)
            plt.imshow(picture)
            plt.show()


Application("img/array10/", "img/image/").run()
print("lool")
