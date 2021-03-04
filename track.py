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
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from itertools import tee


defaultHeight = 480
defaultWidth  = 640

def sign(a):
	if a > 0:
		return 1
	elif a < 0:
		return -1
	else:
		return 0

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class Map():
	def __init__(self, map):
		self.map = map
		self.upperBorder = 5700
		self.bottomBorder = 1000
		self.map[self.map <= self.bottomBorder] = 0
		self.map[self.map >= self.upperBorder] = 0
	
	def cropWindowPart(self, xstart, xend):
		return self.map[0:self.map.shape[0], xstart:xend]

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

	def showFrame(self):
		cv.imshow("frame", self.img)


class MapFrameType():
	def __init__(self, date, path):
		self.date = date
		self.pathImg = path + "/image/"
		self.pathArr = path + "/array10/"
		self.frame = None
		self.map = None
		for i in os.listdir(self.pathImg):
			if i.split(".")[0] == self.date:
				self.frame = Frame(cv.imread(self.pathImg+i))
		for i in os.listdir(self.pathArr):
			if i.split(".")[0] == self.date:
				self.map = Map(np.loadtxt(self.pathArr+i,skiprows = 0))


class Matcher():
	def __init__(self):
 		self.bf = cv.BFMatcher(cv.NORM_HAMMING)
 		self.delta = 0
 		self.perfectMatches = []

 		#kp - key points
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
 		
 		#make own wrapper?...[x1,y1,x2,y2]
 		self.contourCoor = None
 		self.deltaVect = [0,0]

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
		for mat in self.perfectMatches:
			img1_idx = mat.queryIdx
			img2_idx = mat.trainIdx
			(x1,y1) = self.kp1[img1_idx].pt
			(x2,y2) = self.kp2[img2_idx].pt
			cv.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
			cv.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
			cv.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255,0,0), 1)
		cv.line(out, (640*2 + int(self.delta), 0), (640*2 + int(self.delta), 480), (0, 255, 0), thickness=2)
		cv.imshow('Matched Features', out)
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
		goodMathcesDescriptors = []
		retkeypoints = []

		for m,n in matches:
			if m.distance < 0.75*n.distance:
				goodMathcesDescriptors.append(m)
		
		self.kp1 = features1[0]
		self.kp2 = features2[0]

		#trshf - threshold for features - определяет те фичи, котрые не сильно изменились (коэфицент)
		trshf = 5

		sumSpeed = 0
		countOfChangedPoints = 0
		directionSum = 0
		
		deltaVectori = (0,0)
		vectorsSum = [0,0]
		#need to rewrite alghorithm
		for mat in goodMathcesDescriptors:
			img1_idx = mat.queryIdx
			img2_idx = mat.trainIdx

			(x1,y1) = self.kp1[img1_idx].pt
			(x2,y2) = self.kp2[img2_idx].pt
			
			if abs(x2 - x1) < trshf and abs(y2 - y1) < trshf:
				continue
			
			deltaVectori = (x2-x1, y2-y1)
			vectorsSum[0] += deltaVectori[0]
			vectorsSum[1] += deltaVectori[1]

			self.perfectMatches.append(mat)
			pathVectorLen = math.sqrt( (x2 - x1) * (x2-x1) + (y2-y1)*(y2-y1))
			sumSpeed = sumSpeed + pathVectorLen
			countOfChangedPoints = countOfChangedPoints + 1
			directionSum = directionSum + sign(x2-x1)

		self.deltaVect[0] = vectorsSum[0]/countOfChangedPoints
		self.deltaVect[1] = vectorsSum[1]/countOfChangedPoints

		self.delta = int(sumSpeed/countOfChangedPoints) * sign(directionSum)
		return self.perfectMatches
	
	def findCountoursOftruck(self):
		diff = cv.absdiff(self.img1, self.img2)
		blur = cv.GaussianBlur(diff, (5, 5), 0)
		_, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
		dilated = cv.dilate(thresh, None, iterations=3)
		сontours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		good_contours = []

		lowLevelArea = 20000

		upperLevelArea = 100000
		#ADD SORTINQ AND FILTERING PLS
		for contour in сontours:
			(x, y, w, h) = cv.boundingRect(contour)
			if  cv.contourArea(contour) < lowLevelArea or cv.contourArea(contour) > upperLevelArea:  
				continue
			good_contours.append(contour)
			#cv.rectangle(self.img2, (x, y), (x + w, y + h), (0, 255, 0), 2)  
			self.contourCoor = [x, y, x + w, y + h]	

	def calcMapShift(self):
		if self.delta > 0:
			self.sideWhereMoveingFrom = 0
		elif self.delta < 0:
			self.sideWhereMoveingFrom = 1
		else:
			self.sideWhereMoveingFrom = -1

		if self.sideWhereMoveingFrom == 1:
			self.mapShift = Map(self.map2.map[ 0:self.map2.map.shape[0], self.map2.map.shape[1] + self.delta:self.map2.map.shape[1] ])
		elif self.sideWhereMoveingFrom	 == 0:
			self.mapShift = Map(self.map1.map[0:self.map1.map.shape[0], 0:self.delta])

	def shiftCountourCoordinates(self):
		resCoor = []
		self.contourCoor[0] += self.deltaVect[0]
		self.contourCoor[1] += self.deltaVect[1]
		self.contourCoor[2] += self.deltaVect[0]
		self.contourCoor[3] += self.deltaVect[1]

class Truck():
	def __init__(self):
		self.pieces = []
		self.side = -1
		self.truckMatrice = None

	def initPiece(self, p):
		self.pieces.append(p)
		
	def addPiece(self, p):
		self.pieces.append(p)
	
	def update(self, map, delta, side):
		for pi in self.pieces:
			pi.update(delta, map, side)

	def showTruckPiecesbyItsParts(self):
		for pi in self.pieces:
			for i in pi.otherViews:
				plt.imshow(i)
				plt.show()

	def showTruck(self):
		plt.imshow(self.truckMatrice)
		plt.show()

	def assemblyTruck(self):
		sums = []
		print(self.side)
		self.truckLen = 1100
		truckMatrice = np.ndarray(shape=(defaultHeight, self.truckLen), dtype=float, order='F')
		constructed = sumV = self.pieces[0].otherViews[0]
		for pi in range(len(self.pieces)):
			sumV = self.pieces[pi].otherViews[0]
			sumOffset = self.pieces[pi].otherViews[0].shape[1]
			for vi in range(len(self.pieces[pi].otherViews)):
				if vi + 1 >= len(self.pieces[pi].otherViews):
					continue
				if self.side == 1:
					if self.pieces[pi].otherViews[vi+1].shape[1] > sumV.shape[1]:
						continue
					sumV[0:sumV.shape[0], sumV.shape[1]-self.pieces[pi].otherViews[vi+1].shape[1]:sumV.shape[1]] += self.pieces[pi].otherViews[vi+1]
				if self.side == 0:
					if self.pieces[pi].otherViews[vi+1].shape[1] > sumV.shape[1]:
						continue
					sumV[0:sumV.shape[0], 0:self.pieces[pi].otherViews[vi+1].shape[1]] += self.pieces[pi].otherViews[vi+1]
			sums.append(sumV)
			
		if self.side == 1:
			prev = 0
		elif self.side == 0:
			prev = truckMatrice.shape[1]
		for sumi in sums:
			if self.side == 1:
				truckMatrice[0:truckMatrice.shape[0], prev:prev + sumi.shape[1]] = sumi
				prev = prev + sumi.shape[1]
			elif self.side == 0:
				truckMatrice[0:truckMatrice.shape[0], prev - sumi.shape[1]:prev] = sumi
				prev = prev - sumi.shape[1]
		
		self.truckMatrice = truckMatrice
		blurred = median_filter(truckMatrice, size = 8)


class Piece():
	def __init__(self):
		self.x = -1
		self.length = -1
		self.otherViews = []

	def start(self, x, len, view, side):
		if side == 1:
			self.x = x
			self.length = len
			self.otherViews.append(view.cropWindowPart(self.x, view.map.shape[1]))
		if side == 0:
			self.x = x
			self.length = len
			self.otherViews.append(view.cropWindowPart(0, self.x))	
		
	def update(self, delta, map, side):
		if side == 1:
			self.x = self.x + delta
			
			if self.x < 0:
				self.x = 0
				self.length = self.length + delta
			
			crop = map.cropWindowPart(self.x, self.x + self.length)
			self.otherViews.append(crop)
		
		if side == 0:
			self.x = self.x + delta
			if self.x + self.length > map.map.shape[1]:
				self.length = map.map.shape[1] - self.x
			crop = map.cropWindowPart(self.x, self.x + self.length)
			self.otherViews.append(crop)


def app(n, offset):
	c = 0
	imagesList = []
	for i in os.listdir("img/image/"):
		if c > n:
			break

		c = c + 1
		if c < offset:
			continue
		im = MapFrameType(i.split(".")[0], "img/")
		imagesList.append(im)
	framePairs = []
	for i in range(len(imagesList)):
		if i + 1 >= len(imagesList):
			break
		framePairs.append((imagesList[i], imagesList[i+1]))
	initFrames = True
	
	prevCoor = None
	truck = Truck()

	for pair in framePairs:
		matcher = Matcher()
		matcher.compareTwoFrame(pair[0], pair[1])
		
		if initFrames:
			matcher.findCountoursOftruck()
			matcher.calcMapShift()
			piece = Piece()
			if matcher.sideWhereMoveingFrom == 1:
				piece.start(matcher.contourCoor[0], matcher.map2.map.shape[1] - matcher.contourCoor[0], matcher.map2, matcher.sideWhereMoveingFrom)
			elif matcher.sideWhereMoveingFrom == 0:
				piece.start(matcher.contourCoor[1], abs(matcher.delta), matcher.map2, matcher.sideWhereMoveingFrom)
			
			truck.initPiece(piece)
			truck.side = matcher.sideWhereMoveingFrom
			initFrames = False
			continue
		else:
			matcher.contourCoor = prevCoor
		
		matcher.calcMapShift()
		piece = Piece()
		if matcher.sideWhereMoveingFrom == 1:
			piece.start(matcher.map2.map.shape[1] + matcher.delta, abs(matcher.delta), matcher.map2, matcher.sideWhereMoveingFrom)
		elif matcher.sideWhereMoveingFrom == 0:
			piece.start(0 + matcher.delta, abs(matcher.delta), matcher.map2, matcher.sideWhereMoveingFrom)
			
		truck.update(matcher.map2, matcher.delta, matcher.sideWhereMoveingFrom)
		truck.addPiece(piece)
		prevCoor = matcher.contourCoor
	print("END")
	truck.assemblyTruck()
	#truck.showTruckPiecesbyItsParts()
	truck.showTruck()
shadow = app(13, 0)

shadow = app(37, 15)

plt.show()
cv.waitKey(0)
print("lol")
