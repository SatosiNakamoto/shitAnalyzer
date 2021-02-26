import numpy as np
from matplotlib import pyplot as plt
import os
import cv2 as cv
from scipy.ndimage.filters import gaussian_filter, median_filter
import math
from skimage.filters import median, gaussian, threshold_otsu, sobel
from skimage.morphology import binary_erosion
upperBorder = 5700 #5700
bottomBorder = 1000

from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage import data
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage import measure
from PIL import Image

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


class MapFrameType():
	def __init__(self, date, pathImg = None, pathArr = None):
		self.date = date
		self.pathImg = pathImg
		self.pathArr = pathArr
		self.img = None
		self.map = None
		
		for i in os.listdir(self.pathImg):
			if i.split(".")[0] == self.date:
				self.img = Frame(cv.imread(self.pathImg+i))
		for i in os.listdir(self.pathArr):
			if i.split(".")[0] == self.date:
				self.map = Map(np.loadtxt(self.pathArr+i,skiprows = 0))
	
	def prepareFeatures(self):
		self.map.setFeaturePoints(self.img.getFeaturesPoints())

class Map():
	def __init__(self, map):
		self.map = map
		
	#works strange.................
	def setFeaturePoints(self,points):
		self.map[300, 300] = np.nan
		print(self.map[300, 300])
		print("######################")
		for p in points:
			if p[0] >= 480 or p[1] >= 640:
				continue
			print(str(int(p[0])) + " " + str(int(p[1])))
			self.map[int(p[0]),int(p[1])] = np.nan
		print("######################")

	def showMap(self):
		plt.imshow(self.map)

class Frame():
	def __init__(self, img):
		self.img = img
		self.orb = cv.ORB_create()

	def getFeaturesPoints(self):
		if self.img is not None:
			gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
			dst = cv.cornerHarris(gray,2,3,0.099)
			feats = cv.goodFeaturesToTrack(np.mean(self.img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
			kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
			kps, des = self.orb.compute(self.img, kps)
			return (kps, des)

	def drawFeaturesPoints(self, points = None):
		if points == None:
			print("None")
			points = self.getFeaturesPoints()[0]
		#print(kps)
		for p in points:
			#print(p.pt[0])
			if p.pt[1] > 480 or p.pt[0] > 640:
				continue
			cv.circle(self.img, (int(p.pt[0]),int(p.pt[1])),5,(0), 2)		

	def showFrame(self):
		cv.imshow("frame", self.img)

class mapProcessor():
	def __init__(self, p):
		self.path = p
		self.upperBorder = 5700 #5700
		self.bottomBorder = 1000
		self.processedMaps = []

	def getMaps(self,n):
		imgs = []
		c = 0
		for i in os.listdir(self.path):
			if c > n:
				break
			c = c + 1

			img = np.loadtxt(self.path+i,skiprows = 0)

			imgs.append(Map(img))
		return imgs		
	def processNMaps(self,n):
		for map in self.getMaps(n):
			map.map[map.map <= self.bottomBorder] = 0
			map.map[map.map >= self.upperBorder] = 0
			self.processedMaps.append(map)
	
	#0 for left
	#1 for right
	#works only when its first frame of truck
	def disaideWhichHalfIsCloserToCar(self):
		firstMap = self.processedMaps[0].map

		halfOfColumns = int(firstMap.shape[1]/2)
		sumLeftHalf=firstMap.sum(axis=0)
		sumRightHalf=firstMap.sum(axis=0)
		leftHalf = sumLeftHalf[halfOfColumns]
		rightHalf = firstMap.sum() - leftHalf
		if leftHalf > rightHalf:
			return 0
		else:
			return 1

	#use time to split them and disaideWhichHalfIsCloserToCar function
	def splitCarsByItsMaps(self):
		return None


class Matcher():
	def __init__(self):
 		self.bf = cv.BFMatcher(cv.NORM_HAMMING)
 		self.last = None
 		self.speed = 0
 		self.perfectMatches = []
 		self.kp1 = []
 		self.kp2 = []
 		self.img1 = None
 		self.img2 = None

	def drawMatches(self):
		#if self.img1 == None or self.img2 == None or self.kp1 == [] or self.kp2 == [] or self.perfectMatches == []:
		#	print("cant draw compare two frames first")
		#	return
		rows1 = self.img1.shape[0]
		cols1 = self.img1.shape[1]
		rows2 = self.img2.shape[0]
		cols2 = self.img2.shape[1]
		
		out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
		out[:rows1,:cols1] = np.dstack([self.img1, self.img1, self.img1])
		out[:rows2,cols1:] = np.dstack([self.img2, self.img2, self.img2])
		sumSpeed = 0
		countOfChangedPoints = 0
		print("-----------------")
		for mat in self.perfectMatches:
			img1_idx = mat.queryIdx
			img2_idx = mat.trainIdx

			(x1,y1) = self.kp1[img1_idx].pt
			(x2,y2) = self.kp2[img2_idx].pt
			print(str(x1) + " " + str(y1) + "                 " + str(x2) + " " + str(y2))
			cv.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
			cv.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
			cv.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255,0,0), 1)
		print("-----------------")
		cv.line(out, (640*2 - int(self.speed), 0), (640*2 - int(self.speed), 480), (0, 255, 0), thickness=2)
		cv.imshow('Matched Features', out)
		#cv.destroyWindow('Matched Features')
		
		return out

	def compareTwoFrame(self,mapFrame1, mapFrame2):
		features1 = mapFrame1.img.getFeaturesPoints()
		features2 = mapFrame2.img.getFeaturesPoints()
		self.img1 = cv.cvtColor(mapFrame1.img.img, cv.COLOR_BGR2GRAY)
		self.img2 = cv.cvtColor(mapFrame2.img.img, cv.COLOR_BGR2GRAY)
		matches = self.bf.knnMatch(features1[1],features2[1], k=2)
		retdescs = []
		retkeypoints = []

		for m,n in matches:
			if m.distance < 0.75*n.distance:
				retdescs.append(m)
		
		self.kp1 = features1[0]
		self.kp2 = features2[0]
		sumSpeed = 0
		countOfChangedPoints = 0
		for mat in retdescs:
			img1_idx = mat.queryIdx
			img2_idx = mat.trainIdx

			(x1,y1) = self.kp1[img1_idx].pt
			(x2,y2) = self.kp2[img2_idx].pt
			
			if abs(x2 - x1) < 5 and abs(y2 - y1) < 5:
				continue

			self.perfectMatches.append(mat)
			pathVectorLen = math.sqrt( (x2 - x1) * (x2-x1) + (y2-y1)*(y2-y1))
			sumSpeed = sumSpeed + pathVectorLen
			countOfChangedPoints = countOfChangedPoints + 1
		return self.perfectMatches


#mp = mapProcessor("img/")
mf1 = MapFrameType("22-02-2021-00-06-42-110409", "img/image/", "img/array10/")
mf2 = MapFrameType("22-02-2021-00-06-42-259410", "img/image/", "img/array10/")

matcher = Matcher()
matchesList = matcher.compareTwoFrame(mf1,mf2)
resPoints = [matcher.kp2[i.queryIdx].pt for i in matcher.perfectMatches]
mapp = Map(np.loadtxt("img/array10/22-02-2021-00-06-42-259410.txt"))
#mapp.setFeaturePoints(resPoints)

mf3 = MapFrameType("22-02-2021-00-06-42-259410", "img/image/", "img/array10/")
#mf3.img.drawFeaturesPoints(resPoints)
mapp.showMap()
matcher.drawMatches()

plt.show()
cv.waitKey(0)

print("lol")
'''

import cv2  # импорт модуля cv2
import numpy as np
import matplotlib.pyplot as plt
import os

k = 0
frame2 = 0
for i in os.listdir("D:/svt/image12/"):
    frame1 = cv2.imread("D:/svt/image12/" + i)
    if k == 0:
        frame2 = frame1
        k = 1
        continue
    diff = cv2.absdiff(frame1, frame2)  # нахождение разницы двух кадров, которая проявляется лишь при изменении одного из них, т.е. с этого момента наша программа реагирует на любое движение.
    #cv2.imshow('1', diff)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # перевод кадров в черно-белую градацию
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # фильтрация лишних контуров
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)  # метод для выделения кромки объекта белым цветом
    dilated = cv2.dilate(thresh, None, iterations=3)  # данный метод противоположен методу erosion(), т.е. эрозии объекта, и расширяет выделенную на предыдущем этапе область
    сontours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # нахождение массива контурных точек
    
    for contour in сontours:
        (x, y, w, h) = cv2.boundingRect(contour)  # преобразование массива из предыдущего этапа в кортеж из четырех координат
            # метод contourArea() по заданным contour точкам, здесь кортежу, вычисляет площадь зафиксированного объекта в каждый момент времени, это можно проверить
        print(cv2.contourArea(contour))
        if cv2.contourArea(contour) < 700:  # условие при котором площадь выделенного объекта меньше 700 px
            continue
        cv2.rectangle(diff, (x, y), (x + w, y + h), (0, 255, 0), 2)  # получение прямоугольника из точек кортежа
        #cv2.putText(frame1, "Status: {}".format("Dvigenie"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)  # вставляем текст

    plt.subplot(4, 3, k)
    plt.imshow(frame1, interpolation='nearest')
    plt.subplot(4, 3, k + 1)
    plt.imshow(diff, interpolation='nearest')
    plt.subplot(4, 3, k + 2)
    plt.imshow(frame2, interpolation='nearest')
    k += 3
    frame2 = frame1
    if k > 10:
        k = 0
        plt.show()

'''