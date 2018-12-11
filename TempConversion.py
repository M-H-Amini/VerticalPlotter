import cv2
import numpy as np

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
    return [center[0]+x[1]*np.cos(x[0]),center[1]+x[1]*np.sin(x[0])]

image=np.zeros((400,600))
image[200,300]=255
point=[200,300]
image[point[0],point[1]]=255
[theta,r]=car2Pol(point,[200,300])
print([theta*180/np.pi,r])
[new_y,new_x]=pol2Car([theta,r],[200,300])
print([new_y,new_x])
cv2.imshow('image',image)
cv2.waitKey(0)