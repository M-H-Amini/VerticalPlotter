import cv2
import numpy as np

im = cv2.imread('Sample6.jpg')
dst = cv2.fastNlMeansDenoisingColored(im,None,10,10,7,21)
cv2.imshow('denoised',dst)
cv2.waitKey(0)
imgray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',imgray)
cv2.waitKey(0)
#ret,thresh = cv2.threshold(imgray,127,255,0)
thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv2.THRESH_BINARY,15,2)
cv2.imshow('thresh',thresh)
cv2.waitKey(0)
#  Finding contours...
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(im2.shape)
#  Sorting contours...
contours = sorted(contours, key=cv2.contourArea, reverse = True)[:10]
cv2.drawContours(im,contours,-1,(0,255,0))
cv2.imshow('contours',im)
cv2.waitKey(0)