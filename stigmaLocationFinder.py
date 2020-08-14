import cv2 
import os
import numpy as np 
from scipy import signal
  
path = '../214MEDIA/'
files = os.listdir(path)
files = sorted(files)
log = open("output.csv", "w",encoding="utf-8")




heights = np.array([])
widths = np.array([])
image_Names = np.array([])
for image in files:
	
	img_rgb = cv2.imread(path + image)

	# Convert it to grayscale 
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 
	  
	# Read the template 
	template = cv2.imread('template.jpg', 0) 
	  
	# Store width and height of template in w and h 
	h, w = template.shape 
	  
	# Perform match operations. 
	res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) 
	
	# Specify a threshold 
	# threshold = 0.9
	  
	# # Store the coordinates of matched area in a numpy array 
	# loc = np.where( res >= threshold)  

	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	# Draw a rectangle around the matched region. 
	# for pt in zip(*loc[::-1]): 
	#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2) 
	  
	# Show the final image with the matched area. 
	# cv2.namedWindow('Detected',cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('image', 1000,1000)
	# cv2.imshow('Detected',img_rgb) 
	max_loc = np.array(max_loc)

	#adjust for template center
	max_loc[0] += h//2
	max_loc[1] += w//2

	image_Names = np.append(image_Names, image)
	heights= np.append(heights, max_loc[0])
	widths = np.append(widths, max_loc[1])
	
	
	
	# print(image, max_loc, max_val)
	# key = cv2.waitKey(0)
	# cv2.destroyAllWindows()

#median filter location values

heights = signal.medfilt(heights, 9)
widths = signal.medfilt(widths,9)


locations = np.stack((widths, heights), axis=-1)


#print to csv
# locations += str(image)
# locations += str(" ")
# locations += str(max_loc)		
# locations += str("\n")
for location in locations:
	
	print(location[0],",",location[1], file = log)



