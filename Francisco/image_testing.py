import cv2
import numpy as np
from sklearn.metrics import mean_squared_error

path = "/home/franciscoAML/Documents/DSIAC/Five_Classes/DSIAC_3_0/Version7/Single_Channel/yolov3/data/DSIACP_Images_FrameNorm_MAD_23/images/cegr01923_0001_1.png"


img = cv2.imread(path)
print(img.shape)
print(img[1])

img2 = cv2.imread(path,0)
print(img.shape)

error = mean_squared_error(img[:,:,1], img2)
print(error)
#print(img[:,:,0] == img2)