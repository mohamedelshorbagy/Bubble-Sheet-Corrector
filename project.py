import cv2 
import glob 
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
from xlwt import *


xlFile = Workbook()
images = []

for img in glob.glob('dataset/train/*.png'):
    images.append(img)

    

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

    
    
image = cv2.imread(images[1])


cropped = image[750:750+650,95:95+1000]



firstSec = cropped[0:95+1000 , 0:300]
secondSec = cropped[0:95+1000 , 325:640]
thirdSec = cropped[0:95+1000 , 645:640+760]

section_img = []

section_img.append(firstSec)
section_img.append(secondSec)
section_img.append(thirdSec)

#ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3 ,4 : 1}
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3 ,4 : 1 , 5 : 1 , 6 : 2 , 7 : 2 , 8 : 3 , 9 : 2 , 10 : 2 , 11 : 2 , 12 : 3 , 13 : 4 , 14 : 1,15 : 2}   
index = 0; 
for img in section_img: 
        
    
    
    
    studentGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(studentGray , (5 , 5) , 0)
    #studentThresh = cv2.threshold(blurred , 0 , 255 , cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    studentAdaptiveThresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    #    Find the Contours 
    cnts = cv2.findContours(studentAdaptiveThresh.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    questionCnts = []
    index+= 1;
        
    sheet = xlFile.add_sheet("Sheet {}".format(index))    
        
        #24 / 3 /2017 ==> Get the Best Results to get the Contours of an Image
        
    #loop Over the Contours
    for c in cnts:
        
    	# compute the bounding box of the contour, then use the
    	# bounding box to derive the aspect ratio
    	(x, y, w, h) = cv2.boundingRect(c)
    	ar = w / float(h)
             
    	# in order to label the contour as a question, region
    	# should be sufficiently wide, sufficiently tall, and
    	# have an aspect ratio approximately equal to 1
    	if w >= 18 and h >= 18 and ar >= 0.8 and ar <= 1.1:
    		questionCnts.append(c)
    
    #for c in questionCnts:
    #    cv2.drawContours(firstSec , [c] , -1 , (0,255,0) , 2)
    #  
    #
    #    
        
    questionCnts = contours.sort_contours(questionCnts,
    	method="top-to-bottom")[0]
    correct = 0
    
    # each question has 5 possible answers, to loop over the
    # question in batches of 5
    for (q, i) in enumerate(np.arange(0, 60, 4)):
    	# sort the contours for the current question from
    	# left to right, then initialize the index of the
    	# bubbled answer
    	cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
    	bubbled = None
    
    	# loop over the sorted contours
    	for (j, c) in enumerate(cnts):
    		# construct a mask that reveals only the current
    		# "bubble" for the question
    		mask = np.zeros(studentAdaptiveThresh.shape, dtype="uint8")
    		cv2.drawContours(mask, [c], -1, 255, -1)
    
    		# apply the mask to the thresholded image, then
    		# count the number of non-zero pixels in the
    		# bubble area
    		mask = cv2.bitwise_and(studentAdaptiveThresh, studentAdaptiveThresh, mask=mask)
    		total = cv2.countNonZero(mask)
    
    		# if the current total has a larger number of total
    		# non-zero pixels, then we are examining the currently
    		# bubbled-in answer
    		if bubbled is None or total > bubbled[0]:
    			bubbled = (total, j)
    
    	
    	sheet.write(q , 0 , "Q {}".format(q+1))
    	sheet.write(q , 1 , bubbled[1])
    	# draw the outline of the correct answer on the test
    #	cv2.drawContours(firstSec, [questionCnts[k]], -1, color, 3)
    



#cv2.imshow('first',cropped)
cv2.imshow('first' , firstSec)
cv2.imshow('second' , secondSec)
cv2.imshow('third' , thirdSec)
print(len(questionCnts))
print('***************')
print('****************')
print(bubbled[1])
print("******************************")
print(q)
print(ANSWER_KEY[q])
xlFile.save('Grade_Train.xls')
cv2.imshow('mask' , mask)
cv2.waitKey(0)
cv2.destroyAllWindows()    
