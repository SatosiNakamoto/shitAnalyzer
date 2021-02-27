import numpy as np
from matplotlib import pyplot as plt
import os
import cv2 as cv
from scipy.ndimage.filters import gaussian_filter, median_filter
import math
from skimage.filters import median, gaussian, threshold_otsu, sobel
from skimage.morphology import binary_erosion
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

upperBorder = 5700 #5700
bottomBorder = 1000
signTrue = lambda a: (a>0) - (a<0)

def sign(a):
	if a > 0:
		return 1
	elif a < 0:
		return -1
	else:
		return 0

class MapFrameType():
	def __init__(self, date, pathImg = None, pathArr = None):
		self.date = date
		self.pathImg = pathImg
		self.pathArr = pathArr
		self.frame = None
		self.map = None
		for i in os.listdir(self.pathImg):
			if i.split(".")[0] == self.date:
				self.frame = Frame(cv.imread(self.pathImg+i))
		for i in os.listdir(self.pathArr):
			if i.split(".")[0] == self.date:
				self.map = Map(np.loadtxt(self.pathArr+i,skiprows = 0))
	
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
		imageSegment = cv.pyrMeanShiftFiltering(self.frame.img, imageSegment, spatialRadius, colorRadius, pyramidLevels);
		#cv.imshow("segmented",imageSegment)
		#cv.waitKey(0)
		return imageSegment
		
class Map():
	def __init__(self, map):
		self.map = map
		self.upperBorder = 5700 #5700
		self.bottomBorder = 1000
		self.processMap()

	def processMap(self):
		self.map[self.map <= self.bottomBorder] = 0
		self.map[self.map >= self.upperBorder] = 0
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


	#generalize
	def cropShiftPart(self, xstart, xend = None):
		xend = self.map.shape[1]
		return self.map[0:self.map.shape[0], xstart:xend]

	#0 for left
	#1 for right
	#works only when its first frame of truck
	#use time to split them and disaideWhichHalfIsCloserToCar function
	def disaideWhichHalfIsCloserToCar(self):
		halfOfColumns = int(self.map.shape[1]/2)
		window = self.map[0:self.map.shape[1], 0:halfOfColumns]
		leftHalf = window.sum()
		rightHalf = self.map.sum() - leftHalf
		print("right half - " + str(rightHalf))
		print("left half  - " + str(leftHalf))
		
		if leftHalf > rightHalf:
			return 0
		else:
			return 1

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
 		self.mapShift = None
 		#0 - from left; 1 - from right
 		self.sideWhereMoveingFrom = -1
 		#make own wrapper...
 		self.contourCoor = None

	def drawMatches(self):
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
		cv.line(out, (640*2 + int(self.delta), 0), (640*2 + int(self.delta), 480), (0, 255, 0), thickness=2)
		cv.imshow('Matched Features', out)
		#cv.destroyWindow('Matched Features')
		
		return out

	def compareTwoFrame(self,mapFrame1, mapFrame2):
		features1 = mapFrame1.frame.getFeaturesPoints()
		features2 = mapFrame2.frame.getFeaturesPoints()
		self.img1 = cv.cvtColor(mapFrame1.frame.img, cv.COLOR_BGR2GRAY)
		self.img2 = cv.cvtColor(mapFrame2.frame.img, cv.COLOR_BGR2GRAY)
		self.map1 = mapFrame1.map
		self.frame1 = mapFrame1.frame
		self.map2 = mapFrame2.map
		self.frame2 = mapFrame2.frame
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
		directionSum = 0
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
			directionSum = directionSum + sign(x2-x1)
		self.delta = int(sumSpeed/countOfChangedPoints) * sign(directionSum)
		print("delta is" + str(self.delta))
		return self.perfectMatches
	
	def findCountoursOftruck(self):
		diff = cv.absdiff(self.img1, self.img2)
		blur = cv.GaussianBlur(diff, (5, 5), 0)
		_, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
		dilated = cv.dilate(thresh, None, iterations=3)
		сontours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		print("cont len is" + str(len(сontours)))
		good_cont = []

		#ADD SORTINQ AND FILTERING PLS
		for contour in сontours:
			(x, y, w, h) = cv.boundingRect(contour)
			print(cv.contourArea(contour))
			if  cv.contourArea(contour) < 20000 or cv.contourArea(contour) > 100000:  
				continue
			good_cont.append(contour)
			cv.rectangle(self.img2, (x, y), (x + w, y + h), (0, 255, 0), 2)  
			self.contourCoor = [x, y, x + w, y + h]
			#cv.drawContours(self.img2, good_cont, -1, (0, 255, 0), 2)
		#cv.imshow("contours", self.img2)
		#cv.waitKey(0)
	
	#def findFirstMapMask(self):
	

	def calcMapShift(self):
		if self.delta > 0:
			self.sideWhereMoveingFrom = 0
		elif self.delta < 0:
			self.sideWhereMoveingFrom = 1
		else:
			self.sideWhereMoveingFrom = -1

		print("side is" + str(self.sideWhereMoveingFrom))
		print("delta is " +  str(self.delta))
		if self.sideWhereMoveingFrom == 1:
			self.mapShift = Map(self.map2.map[ 0:self.map2.map.shape[0], self.map2.map.shape[1] + self.delta:self.map2.map.shape[1] ])
		elif self.sideWhereMoveingFrom	 == 0:
			self.mapShift = Map(self.map1.map[0:self.map1.map.shape[0], 0:self.delta])

	def shiftCountourCoordinates(self):
		print("before" + str(self.contourCoor))
		resCoor = []
		for i in self.contourCoor:
			i = i + self.delta
			resCoor.append(i)
		self.contourCoor = resCoor
		print("after" + str(self.contourCoor))

#shadow of truck constructed from frames and its maps
class Shadow():
	def __init__(self):
		self.shadowNodesList = []


class ShadowNode():
	def __init__(self):
		self.pointerToNextShadowNode = None
		self.nodeMap = None
		self.nodeFrame = None
		

def printContrs(n):
	c = 0
	imagesList = []
	for i in os.listdir("img/image/"):
		if c > n:
			break
		c = c + 1

		print("img/images/" + i)
		im = MapFrameType(i.split(".")[0], "img/image/", "img/array10/")
		imagesList.append(im)
		#cv.imshow("i",i)
		#cv.waitKey(0)
	framePairs = []
	for i in range(len(imagesList)):
		if i + 1 >= len(imagesList):
			break
		framePairs.append((imagesList[i], imagesList[i+1]))
	initFrames = True
	#отсюда можно найти длину грузовика (не самый лучший способ, как по мне) но мне ооочень не нраваится то, как это написано!!!ужас
	prevCoor = None
	shadow = Shadow()
	for pair in framePairs:
		matcher = Matcher()

		matcher.compareTwoFrame(pair[0], pair[1])
		if initFrames:
			matcher.findCountoursOftruck()
			initFrames = False
			shadow.shadowNodesList.append(Map(matcher.map1.cropShiftPart(matcher.contourCoor[0])))
		else:
			matcher.contourCoor = prevCoor
		#matcher.drawMatches()
		matcher.calcMapShift()
		matcher.shiftCountourCoordinates()
		shadow.shadowNodesList.append(matcher.mapShift)
		prevCoor = matcher.contourCoor
		#matcher.mapShift.showMap()
		#plt.show()
		#cv.waitKey(0)
	for s in shadow.shadowNodesList:
		plt.imshow(s.map)
		plt.show()


#mp = mapProcessor("img/")
printContrs(15)
mf1 = MapFrameType("22-02-2021-00-06-42-259410", "img/image/", "img/array10/")
mf2 = MapFrameType("22-02-2021-00-06-42-389983", "img/image/", "img/array10/")
matcher = Matcher()
matchesList = matcher.compareTwoFrame(mf1,mf2)
cv.waitKey(0)

matcher.findCountoursOftruck()
matcher.drawMatches()
matcher.calcMapShift()
matcher.mapShift.showMap()
print(matcher.contourCoor)
plt.show()
cv.waitKey(0)
print("lol")
