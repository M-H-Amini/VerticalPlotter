import cv2
import numpy as np

im = cv2.imread('Sample5.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',imgray)
cv2.waitKey(0)
ret,thresh = cv2.threshold(imgray,127,255,0)
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