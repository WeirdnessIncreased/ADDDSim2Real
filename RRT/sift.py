import cv2
import numpy as np

img = cv2.imread('1.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#sift特征计算
sift = cv2.SIFT_create()

#detect方法找到关键点
kp,res = sift.detectAndCompute(gray,None)

#drawKeypoints方法绘制关键点
img=cv2.drawKeypoints(gray,kp,img)

cv2.imshow('Sift',img)
cv2.waitKey(0)

