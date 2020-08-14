import cv2 
import os
import numpy as np 
from numpy import genfromtxt
from scipy import signal
from skimage import img_as_float64, img_as_ubyte




def display_image(image):
	cv2.namedWindow('stigma',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('stigma', 1000,1000)
	cv2.imshow('stigma', image)
	key = cv2.waitKey(0)
	cv2.destroyAllWindows() 


path = '../214MEDIA/'
files = os.listdir(path)
files = sorted(files)
log = open("beeOrNoBee.csv", "w",encoding="utf-8")

stigma_locations = np.array(genfromtxt('output.csv', delimiter=','), dtype=int)


template = cv2.imread('template.jpg', 0) 
h, w = template.shape

for i in range(430, len(files) - 15):
	image = cv2.imread(path + files[i])
	img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	stigma = img_gray[ (stigma_locations[i][0] - h//2): (stigma_locations[i][0] + h//2), \
		(stigma_locations[i][1] - w//2): (stigma_locations[i][1] + w//2)]
	stigma = img_as_float64(stigma)


	#make average image
	average = np.zeros((h,w), dtype=np.float64)

	#adjust temporal window with n
	n = 101

	for j in range(n):

		#read in image
		current_avg = cv2.imread(path + files[i + j - (n//2)])

		#convert to grayscale
		img_gray_avg = cv2.cvtColor(current_avg, cv2.COLOR_BGR2GRAY) 

		#read in stigma location to center image around
		stig_loca = stigma_locations[i + j - (n//2)]

		#center image on stigma
		current_image = img_gray_avg[ (stig_loca[0] - h//2): (stig_loca[0] + h//2), \
			(stig_loca[1] - w//2): (stig_loca[1] + w//2)]
		
		#convert to float
		current_image = img_as_float64(current_image)

		average = average + current_image
	
	average = average / n
	average = img_as_ubyte(average)
	stigma = img_as_ubyte(stigma)
	difference = abs(average - stigma)
	
	display_image(difference)

	