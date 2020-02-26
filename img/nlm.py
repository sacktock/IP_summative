import cv2
import numpy

##############################################################

# These parameter values are indicative. You should choose your own 
# according to properties of the method you want to demonstrate

h = 45
templateWindowSize = 11
searchWindowSize = 35

# noise uses h = 3, h = 5, h = 10 and 7 x 7, and 21 x 21
# hish noise uses h = 10, h = 13, h = 15 and 11 x 11 and 35 x 35
# extreme noise uses h = 25, h = 35, h = 45 and 11 x 11 and 35 x 35
##############################################################

img1 = cv2.imread('denoised-Noise.png')
img2 = cv2.imread('denoised-highNoise.png')
img3 = cv2.imread('denoised-extremeNoise.png')
#dst = cv2.GaussianBlur(img, (11,11), 0)
vis = numpy.concatenate((img1, img2, img3), axis=0)
#img = cv2.imread('dice-extremeNoise.png')
#dst = cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWindowSize, 21)
cv2.imwrite('denoised.png', vis) 


