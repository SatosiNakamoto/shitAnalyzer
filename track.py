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

upperBorder = 5700 #5700
bottomBorder = 1000
signTrue = lambda a: (a>0) - (a<0)
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
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


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
		#print(self.map[300, 300])
		#print("######################")
		for p in points:
			if p[0] >= 480 or p[1] >= 640:
				continue
			#print(str(int(p[0])) + " " + str(int(p[1])))
			self.map[int(p[0]),int(p[1])] = np.nan
		#print("######################")


	#generalize
	def cropShiftPart(self, xstart, xend = None):
		xend = self.map.shape[1]
		return self.map[0:self.map.shape[0], xstart:xend]
	def cropWindowPart(self, xstart, xend):
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
		#print("right half - " + str(rightHalf))
		#print("left half  - " + str(leftHalf))
		
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
		#print("-----------------")
		for mat in self.perfectMatches:
			img1_idx = mat.queryIdx
			img2_idx = mat.trainIdx

			(x1,y1) = self.kp1[img1_idx].pt
			(x2,y2) = self.kp2[img2_idx].pt
			#print(str(x1) + " " + str(y1) + "                 " + str(x2) + " " + str(y2))
			cv.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
			cv.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
			cv.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255,0,0), 1)
		#print("-----------------")
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

		#ADD SORTINQ AND FILTERING PLS
		for contour in сontours:
			(x, y, w, h) = cv.boundingRect(contour)
			#print(cv.contourArea(contour))
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

		#print("side is" + str(self.sideWhereMoveingFrom))
		#print("delta is " +  str(self.delta))
		if self.sideWhereMoveingFrom == 1:
			self.mapShift = Map(self.map2.map[ 0:self.map2.map.shape[0], self.map2.map.shape[1] + self.delta:self.map2.map.shape[1] ])
		elif self.sideWhereMoveingFrom	 == 0:
			self.mapShift = Map(self.map1.map[0:self.map1.map.shape[0], 0:self.delta])

	def shiftCountourCoordinates(self):
		#print("before" + str(self.contourCoor))
		resCoor = []
		for i in self.contourCoor:
			i = i + self.delta
			resCoor.append(i)
		self.contourCoor = resCoor
		#print("after" + str(self.contourCoor))


#shadow of truck constructed from frames and its maps
class Shadow():
	def __init__(self):
		self.shadowNodesList = []

	def showShadow(self):
		c = 0
		for shd in self.shadowNodesList:
			print("view number " + str(c) + " len is " + str(len(shd.otherViews)))
			#plt.imshow(shd.originalMap.map)
			#plt.show()
			c = c + 1
			for i in shd.otherViews:
				plt.imshow(i)
				plt.show()


			#plt.plot(range(len(shd.otherViews)))
			#plt.figure()
			#for i in shd.otherViews:
			#	plt.imshow(i)
			#plt.show()
			#for view in shd.otherViews:
				#plt.plot(range(10))
				#plt.figure()
				#plt.plot(range(10), 'ro-')
				#plt.imshow(view)
				#print("view number " + str(c) + " len is " + len(shd.otherViews))
				#c = c + 1
			#plt.show()

class ShadowNode():
	def __init__(self):
		self.otherViews = []
		self.nodeMap = None
		self.nodeFrame = None
		self.shiftValue = None
		self.originalMap = None
		
	def setShadowNodeMap(self, map):
		self.nodeMap = map
	def setOriginalMap(self, m):
		self.originalMap = m

	def helpPreviousShadow(self, deltas, num):
		coors = []
		sum = 0
		for d in reversed(deltas[:num + 1]):
			sum = sum + d
			coors.append(sum)
		parts = []
		if len(coors) > 1:
			coors.insert(0, 0)
		#print("soors is " + str(coors))

		for c1, c2 in pairwise(coors):
			startWindow = self.originalMap.map.shape[1] + c2
			endWindow   = self.originalMap.map.shape[1] + c1
			if startWindow < self.originalMap.map.shape[1] and endWindow < self.originalMap.map.shape[1] and startWindow > 0 and endWindow > 0:#не круто, например первый фрейм просто вылетит 
				parts.append(self.originalMap.cropWindowPart(startWindow, endWindow))
			
		return parts


def printContrs(n):
	c = 0
	imagesList = []
	for i in os.listdir("img/image/"):
		if c > n:
			break
		c = c + 1

		#print("img/images/" + i)
		im = MapFrameType(i.split(".")[0], "img/image/", "img/array10/")
		imagesList.append(im)
	framePairs = []
	for i in range(len(imagesList)):
		if i + 1 >= len(imagesList):
			break
		framePairs.append((imagesList[i], imagesList[i+1]))
	initFrames = True
	
	#отсюда можно найти длину грузовика (не самый лучший способ, как по мне) но мне ооочень не нраваится то, как это написано!!!ужас
	prevCoor = None
	shadow = Shadow()
	deltas = []
	for pair in framePairs:
		matcher = Matcher()

		matcher.compareTwoFrame(pair[0], pair[1])
		if initFrames:
			matcher.findCountoursOftruck()
			deltas.append(matcher.map2.map.shape[1] - matcher.contourCoor[0])
			#plt.imshow(matcher.map2.map)
			#plt.show()
			shadowNodeAppend = ShadowNode()
			shadowNodeAppend.setOriginalMap(matcher.map2)
			shadowNodeAppend.setShadowNodeMap(Map(matcher.map1.cropShiftPart(matcher.contourCoor[0])))
			shadow.shadowNodesList.append(shadowNodeAppend)
			initFrames = False
		else:
			matcher.contourCoor = prevCoor
		#plt.imshow(matcher.map2.map)
		#plt.show()
		matcher.calcMapShift()
		deltas.append(matcher.delta)
		print(deltas)
		matcher.shiftCountourCoordinates()
		shadowNodeAppend = ShadowNode()
		shadowNodeAppend.setOriginalMap(matcher.map2)
		shadowNodeAppend.setShadowNodeMap(matcher.mapShift)
		shadow.shadowNodesList.append(shadowNodeAppend)
		prevCoor = matcher.contourCoor

	deltas[0] = deltas[0] * sign(deltas[1])
	for mi in range(len(shadow.shadowNodesList)):
		shadow.shadowNodesList[mi].shiftValue = deltas[mi]

	numberOfShadow = 0
	for shd in shadow.shadowNodesList:
		helperMaps = list(reversed(shd.helpPreviousShadow(deltas, numberOfShadow)))
		numberOfShadow = numberOfShadow + 1
		
		for i in helperMaps:
			plt.imshow(i)
			plt.show()
		print("+++++++++++++++++++++++++++++++++++++++")
		
		for i in range(len(helperMaps)):
			shadow.shadowNodesList[i].otherViews.append(helperMaps[i])#не то добавляешь ни туда
	return shadow

class Truck():
	def __init__(self):
		self.pieces = []

	def initPiece(self, p):
		self.pieces.append(p)
		
	def addPiece(self, p):
		self.pieces.append(p)
	
	def update(self, map, delta):
		for pi in self.pieces:
			pi.update(delta, map)

	def showTruck(self):
		for pi in self.pieces:
			print(len(pi.otherViews))
			for i in pi.otherViews:
				plt.imshow(i)
				plt.show()

	def assemblyTruck(self):
		sums = []
		self.truckLen = 1200
		truckMatrice = np.ndarray(shape=(defaultHeight, self.truckLen), dtype=float, order='F')
		constructed = sumV = self.pieces[0].otherViews[0]
		for pi in range(len(self.pieces)):
			sumV = self.pieces[pi].otherViews[0]
			#print("++++++++++++++++")
			sumOffset = self.pieces[pi].otherViews[0].shape[1]
			for vi in range(len(self.pieces[pi].otherViews)):
				if vi + 1 >= len(self.pieces[pi].otherViews):
					#print("skeep")
					continue
				#print("sizes ", sumV.shape[1]-self.pieces[pi].otherViews[vi+1].shape[1],sumV.shape[1])
			
				#plt.figure(1)
				#plt.imshow(sumV[0:sumV.shape[0], sumV.shape[1]-self.pieces[pi].otherViews[vi+1].shape[1]:sumV.shape[1]])
				#plt.show()
				#plt.figure(2)
				#plt.imshow(self.pieces[pi].otherViews[vi+1])
				#plt.show()
				if self.pieces[pi].otherViews[vi+1].shape[1] > sumV.shape[1]:
					continue

				sumV[0:sumV.shape[0], sumV.shape[1]-self.pieces[pi].otherViews[vi+1].shape[1]:sumV.shape[1]] += self.pieces[pi].otherViews[vi+1]
				
				#plt.figure(3)
				#print("sum is")
				#plt.imshow(sumV)
				#plt.show()
			#print("----------")
			#plt.imshow(sumV)
			#plt.show()
			sums.append(sumV)
			#plt.imshow(sumV)
			#plt.show()
			#print("NODE END")

		prev = 0
		for sumi in sums:
			#plt.imshow(sumi)
			#plt.show()
			print("prev: ",prev, " shape: ", sumi.shape[1])
			truckMatrice[0:truckMatrice.shape[0], prev:prev + sumi.shape[1]] = sumi
			prev =prev +  sumi.shape[1]
		blurred = median_filter(truckMatrice, size = 8)

		plt.imshow(truckMatrice)
		plt.show()
		plt.imshow(blurred)
		plt.show()

class Piece():
	def __init__(self):
		self.x = -1
		self.length = -1
		self.otherViews = []

	def start(self, x, len, view):
		self.x = x
		self.length = len
		self.otherViews.append(view.cropWindowPart(self.x, view.map.shape[1]))
		
	def update(self, delta, map):
		#SIGN
		self.x = self.x + delta
		
		if self.x < 0:
			self.x = 0
			self.length = self.length + delta

		crop = map.cropWindowPart(self.x, self.x + self.length)
		self.otherViews.append(crop)
			

def constructContrs(n):
	c = 0
	imagesList = []
	for i in os.listdir("img/image/"):
		if c > n:
			break
		c = c + 1
		im = MapFrameType(i.split(".")[0], "img/image/", "img/array10/")
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
			piece = Piece()
			piece.start(matcher.contourCoor[0], matcher.map2.map.shape[1] - matcher.contourCoor[0], matcher.map2)
			
			truck.initPiece(piece)
			initFrames = False
			continue
		else:
			matcher.contourCoor = prevCoor
		
		#matcher.calcMapShift()
		#matcher.shiftCountourCoordinates()
		piece = Piece()
		#SIGN?
		#print(matcher.delta)
		piece.start(matcher.map2.map.shape[1] + matcher.delta, abs(matcher.delta), matcher.map2)
			
		truck.update(matcher.map2, matcher.delta)
		truck.addPiece(piece)
		prevCoor = matcher.contourCoor
	truck.assemblyTruck()
	#truck.showTruck()
	
#mp = mapProcessor("img/")
shadow = constructContrs(13)
#shadow.showShadow()

'''
mf1 = MapFrameType("22-02-2021-00-06-42-259410", "img/image/", "img/array10/")
plt.show()
mf2 = MapFrameType("22-02-2021-00-06-42-389983", "img/image/", "img/array10/")
matcher = Matcher()
matchesList = matcher.compareTwoFrame(mf1,mf2)
cv.waitKey(0)

#matcher.findCountoursOftruck()
#matcher.drawMatches()
#matcher.calcMapShift()
#matcher.mapShift.showMap()
#print(matcher.contourCoor)
'''
plt.show()
cv.waitKey(0)
print("lol")
