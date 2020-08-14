import cv2 
import os
import numpy as np 
from numpy import genfromtxt
from scipy import signal
from skimage import img_as_float64, img_as_ubyte




def display_image(image):
	cv2.namedWindow('stigma',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('stigma', 500,500)
	cv2.imshow('stigma', image)
	key = cv2.waitKey(0)
	cv2.destroyAllWindows() 


def average_maker():
	path = '../214MEDIA/'
	files = os.listdir(path)
	files = sorted(files)
	n = len(files)

	image_for_size = cv2.imread(path + files[0])

	average = np.zeros(image_for_size.shape)

	for file in files:
		image = cv2.imread(path + file)
		image = img_as_float64(image) #convert to floats for math
		average = average + image

	average = img_as_ubyte(average / n) # covert back to 8 bit before saving image
	img_gray = cv2.cvtColor(average, cv2.COLOR_BGR2GRAY)
	cv2.imwrite('average.png', img_gray)
	return img_gray


def difference_maker(average, image):
	image = abs(average - image) 

	return image
	


path = '../214MEDIA/'
files = os.listdir(path)
files = sorted(files)

log = open("beeOrNoBee.csv", "w",encoding="utf-8")
stigma_locations = np.array(genfromtxt('output.csv', delimiter=','), dtype=int)
average = cv2.imread('average.png', 0)



template = cv2.imread('template.jpg', 0) 
h, w = template.shape


for i, file in enumerate(files):

	image = cv2.imread(path + file, 0)
	# img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	

	#skip image if stigma is not located
	if stigma_locations[i][0] - 150 > 0 and stigma_locations[i][1] - 150 > 0 and \
		stigma_locations[i][0] + 150 < average.shape[0] and \
		stigma_locations[i][1] + 150 < average.shape[1]:
		
		difference = difference_maker(average, image)[stigma_locations[i][0] - 150 : \
		 	stigma_locations[i][0] + 150 , stigma_locations[i][1] - 150 : \
		 	stigma_locations[i][1] + 150]
	
	#if there is a blotch
		#where is it
		#is it close to stigma
	
		if i > 435:
			display_image(difference)
			print(i)
	# print(files[i],",",beeOrNoBee, file=log)