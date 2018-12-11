# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 00:11:10 2018

@author: MHA
"""
import numpy as np
import cv2

def car2Pol(x,center):
    if x[0]!=center[0]:
        return [np.arctan2((x[1]-center[1]),(x[0]-center[0])),np.sqrt((x[0]-center[0])**2+(x[1]-center[1])**2)]
    else:
        if x[1]!=center[1]:
            #return [-np.pi/2*(x[1]-center[1])/abs(x[1]-center[1]),abs(x[1]-center[1])]
            return [np.pi/2*(x[1]-center[1])/abs(x[1]-center[1]),abs(x[1]-center[1])]
        else:
            return [0,0]

##  Polar 2 Cartesian conversion...
def pol2Car(x,center):
    return [int(center[0]+x[1]*np.cos(x[0])),int(center[1]+x[1]*np.sin(x[0]))]


##  Draws a contour with the center specified...
def drawContour(polar_contours,page_size,scale):
    [m,n] = [int(page_size[0]*scale),int(page_size[1]*scale)]
    image = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            image[i, j] = 255
    for contour in polar_contours:
        points = []
        for q in contour:
            image[int(q[1]), int(q[0])] = 0
            #image[int(q[1]), int(q[0])] = 0
        cv2.imshow('result', image)


def regressor1(point1, point2):
    return [int((point1[0]+point2[0])/2), int((point1[1]+point2[1])/2)]


def regression1(contour):
    points_counter = 0
    temp = contour.copy()
    size = len(contour)
    #print('before {}'.format(len(temp)))
    i = 0
    points_counter = 0
    while i < size:
        if ((temp[i+points_counter][0]-temp[i+points_counter-1][0])**2+(temp[i+points_counter][1]-temp[i+points_counter-1][1])**2)>3:
            #print('Points {} and {} : Distance = {}'.format(temp[i+points_counter],temp[i+points_counter-1],((temp[i+points_counter][0]-temp[i+points_counter-1][0])**2+(temp[i+points_counter][1]-temp[i+points_counter-1][1])**2)))
            point=regressor1(temp[i+points_counter],temp[i+points_counter-1])
            #print(point)
            #print('index {} len {}'.format(i+points_counter,len(temp)))
            temp.insert(i+points_counter,point)
            points_counter+=1
        i+=1
    #print('after {}'.format(len(temp)))
    size = len(temp)
    return len(temp),temp

def contourScale(polar_contour, center, page_size , scale=1):
    [m,n] = [int(page_size[0]*scale),int(page_size[1]*scale)]
    image = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            image[i, j] = 255
    points = []
    #print('CS: page size : {} \n contour_center: {}'.format([m,n],center))
    for q in polar_contour:
        point = pol2Car([q[0],q[1]*scale], center)
        #print('CS: original point: {}'.format(q))
        #print('CS: point: {}'.format(point))
        points.append(point)
        image[int(point[0]), int(point[1])] = 0
    #cv2.imshow('result', image)
    return points

def processContour(im,cnt, contour_center, page_size, scale):
    #  Finding center of a contour ...
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    center=[cx,cy]
    #print('sure {}'.format([cx,cy]))
    page_center_scaled=[int(page_size[0]*scale/2),int(page_size[1]*scale/2)]
    #print('Scaled page center = {}'.format(page_center_scaled))
    imcontours = im.copy()
    cv2.circle(imcontours,(cy,cx),10,(0,255,0))
    cv2.circle(imcontours,(int(page_size[1]/2),int(page_size[0]/2)),10,(255,0,0))
    cv2.drawContours(imcontours, cnt, 1, (0,255,0), 3)
    cv2.imshow('Contours',imcontours)
    cv2.waitKey(0)
    cnt = np.reshape(cnt,(cnt.shape[0],cnt.shape[2]))
    cnt = np.ndarray.tolist(cnt)
    polar_contour = [car2Pol([i,j],[cx,cy]) for [i,j] in cnt]
    #print(contour_center)
    contour_center_cartesian=pol2Car([contour_center[0],contour_center[1]*scale],page_center_scaled)
    #contour_center_cartesian=[contour_center[0]*scale,contour_center[1]*scale]
    polar_contour=contourScale(polar_contour, contour_center_cartesian , page_size, scale)
    #print('Scaled contour center is {}'.format((contour_center_cartesian)))
    contour_len=len(cnt)
    delta_len=len(cnt)
    last_delta_len=10*delta_len
    counter=0
    while(abs(delta_len)>10 and counter<10):
        counter+=1
        last_contour_len = contour_len
        contour_len, polar_contour=regression1(polar_contour)
        delta_len = contour_len - last_contour_len
    drawContour([polar_contour],page_size,scale)
    return polar_contour


im = cv2.imread('Sample5.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
#  Finding contours...
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(im2.shape)
#  Sorting contours...
contours = sorted(contours, key=cv2.contourArea, reverse = True)[:10]
cnt=contours[1]

contour_centers=[]
page_size=[im.shape[0], im.shape[1]]
page_center=[int(page_size[0]/2),int(page_size[1]/2)]
scaled_contours=[]
scale=1.5
contours.remove(contours[0])
for i in range(len(contours)):
    print('******{}*******'.format((i)))
    M = cv2.moments(contours[i])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    center=[cx,cy]
    drawContour(contours[i],page_size,1)
    cv2.waitKey(0)
    #print('Original page center = {}'.format(page_center))
    #print('Original contour center is {}'.format(center))
    theta,r=car2Pol(center,page_center)
    contour_centers.append([theta,r])
    #print('Polar contour center is {}'.format([theta,r]))
    scaled_contours.append(processContour(im,contours[i], contour_centers[i], page_size, scale))
print(contour_centers)
drawContour(scaled_contours,page_size,scale)
cv2.waitKey(0)