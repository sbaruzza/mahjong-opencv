import numpy as np
import cv2
from itertools import izip

def rectify(contour):
	contour = contour.reshape((4, 2))
	newContour = np.zeros((4, 2), dtype=np.float32)
	add = contour.sum(1)
	newContour[0] = contour[np.argmin(add)]
	newContour[2] = contour[np.argmax(add)]

	diff = np.diff(contour, axis=1)
	newContour[1] = contour[np.argmin(diff)]
	newContour[3] = contour[np.argmax(diff)]
	return newContour

def preprocess(image):
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	threshold = cv2.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)
	#blurredTreshold	= cv2.GaussianBlur(threshold, (5, 5), 5)
	return threshold

def imgdiff(img1,img2):
	img1 = cv2.GaussianBlur(img1,(5,5),5)
  	img2 = cv2.GaussianBlur(img1,(5,5),5)    
  	diff = cv2.absdiff(img1,img2)  
  	diff = cv2.GaussianBlur(diff,(5,5),5)    
  	flag, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY) 
  	result = np.sum(diff)  
  	return result

def find_closest_card(training,img):
  	features = preprocess(img)
  	return sorted(training.values(), key=lambda x:imgdiff(x[1], features))[0][0]

im = cv2.imread('bambu3.jpg')
thresh = preprocess(im)
cv2.imwrite('threshold.jpg', thresh)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
tiles = []
for c in contours:
	perimeter = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02*perimeter, True)
	area = cv2.contourArea(approx)
	if area > 800 and area < 10000 and len(approx) == 4:
		tiles.append(approx);

tiles.sort(key=lambda x: x[0][0][0])
index = 0
for tile in tiles:
	temp = im.copy()
	cv2.drawContours(temp, [tile], 0, (0, 255, 0), 3)
	cv2.imwrite("contour_%s.jpg" % index, temp)
	index += 1

index = 0
for tile in tiles:
	rectangle = cv2.minAreaRect(tile)
	r = cv2.cv.BoxPoints(rectangle)
	h = np.array([[0,0], [449, 0], [449,449], [0,449]], np.float32)
	transform = cv2.getPerspectiveTransform(rectify(tile), h)
	warp = cv2.warpPerspective(im, transform, (450, 450))
	cv2.imwrite('warp_%s.jpg'% index, warp)
	index += 1

trained = dict([i+1, tiles[i]] for i in range(0, len(tiles)))

unknown = cv2.imread('bambu_tile5.jpg')
print find_closest_card(trained, unknown)

