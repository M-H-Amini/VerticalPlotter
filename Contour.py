# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 00:11:10 2018

@author: MHA
"""
import numpy as np
import cv2

##  Cartesian 2 Polar conversion...
def car2Pol(x,center):
    if x[0]!=center[0]:
        return [np.arctan2((x[1]-center[1]),(x[0]-center[0])),np.sqrt((x[0]-center[0])**2+(x[1]-center[1])**2)]    
    else:
        return [-np.pi/2*(x[1]-center[1])/abs(x[1]-center[1]),abs(x[1]-center[1])]


##  Polar 2 Cartesian conversion...
def pol2Car(x,center):
    return [center[0]+x[1]*np.cos(x[0]),center[1]+x[1]*np.sin(x[0])]


##  Draws a contour with the center specified...
def drawContour(polar_contour):
    m, n = 700, 700
    image = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            image[i, j] = 255
    points = []
    for q in polar_contour:
        image[int(q[0]), int(q[1])] = 0
    cv2.imshow('result', image)


def regressor1(point1, point2):
    return [int((point1[0]+point2[0])/2), int((point1[1]+point2[1])/2)]


def regression1(contour):
    points_counter = 0
    temp = contour.copy()
    size = len(contour)
    print('before {}'.format(len(temp)))
    i = 0
    points_counter = 0
    while i < size:
        if ((temp[i+points_counter][0]-temp[i+points_counter-1][0])**2+(temp[i+points_counter][1]-temp[i+points_counter-1][1])**2)>2:
            print('Points {} and {} : Distance = {}'.format(temp[i+points_counter],temp[i+points_counter-1],((temp[i+points_counter][0]-temp[i+points_counter-1][0])**2+(temp[i+points_counter][1]-temp[i+points_counter-1][1])**2)))
            point=regressor1(temp[i+points_counter],temp[i+points_counter-1])
            #print(point)
            #print('index {} len {}'.format(i+points_counter,len(temp)))
            temp.insert(i+points_counter,point)
            points_counter+=1
        i+=1
    print('after {}'.format(len(temp)))
    size = len(temp)
    return temp

im = cv2.imread('Sample2.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
#  Finding contours...
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(im2.shape)
#  Sorting contours...
contours = sorted(contours, key=cv2.contourArea, reverse = True)[:10]
cnt=contours[1]

def contourSclae(polar_contour, center, scale=1):
    m, n = 700, 700
    image = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            image[i, j] = 255
    points = []
    for q in polar_contour:
        print('***')
        print(q)
        point = pol2Car([q[0],q[1]*scale], center)
        points.append(point)
        image[int(point[0]), int(point[1])] = 0
    #cv2.imshow('result', image)
    return points

def processContour(cnt):
    #  Finding center of a contour ...
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    center=[cx,cy]
    #print( M )
    #print(cnt)
    imcontours = im.copy()
    cv2.circle(imcontours,(cx,cy),10,(0,255,0))
    cv2.drawContours(imcontours, contours, 1, (0,255,0), 3)
    cv2.imshow('Contours',imcontours)
    cnt = np.reshape(cnt,(cnt.shape[0],cnt.shape[2]))
    cnt = np.ndarray.tolist(cnt)
    polar_contour = [car2Pol([i,j],[cx,cy]) for [i,j] in cnt]
    polar_contour=contourSclae(polar_contour, center, 1)
    polar_contour=regression1(polar_contour)
    polar_contour=regression1(polar_contour)
    polar_contour=regression1(polar_contour)
    drawContour(polar_contour)
    '''
    draw(polar_contour,center)
    cnt=regression1(cnt)
    cnt=regression1(cnt)
    cnt=regression1(cnt)
    cnt=regression1(cnt)
    cnt=regression1(cnt)
    polar_contour = [car2Pol([i,j], [cx,cy]) for [i,j] in cnt]
    draw(polar_contour, center)
    '''
processContour(cnt)
cv2.waitKey(0)
