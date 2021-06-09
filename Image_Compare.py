import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename
import urllib.request
from PIL import Image


percentage_list = []

filename1 = askopenfilename(filetypes=[("image","*.jpg")]) # queryImage
filename2 = askopenfilename(filetypes=[("image","*.jpg")]) # trainImage

img1=cv2.imread(filename1,4)
img2=cv2.imread(filename2,4)

grayImage1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
grayImage2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

(thresh, blackAndWhiteImage1) = cv2.threshold(grayImage1, 127, 255, cv2.THRESH_BINARY)
(thresh, blackAndWhiteImage2) = cv2.threshold(grayImage2, 127, 255, cv2.THRESH_BINARY)

# Initiate SURF detector
sift=cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SURF
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        a=len(good)
        percent=(a*100)/len(kp2)
        percentage_list.append(percent)
			
sim_value = max(percentage_list)
#print(sim_value)
if sim_value >= 1.00:
    print("Match")
else:
    print("Not Match")
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)