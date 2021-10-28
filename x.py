import cv2
import numpy as np
import pytesseract as tess
import matplotlib.pyplot as plt


#while True:....
   # success, img = cv2.imread("1.jpg")
img =cv2.imread('images.jpg')
#plt.imshow(img)
#plt.show()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
contours = cv2.findContours(thresh,1,2)
h = cv2.findContours(thresh,1,2)

largest_rectangle = [0,0]
for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.018*cv2.arcLength(cnt,True),True)
    if len(approx)==4:
        area = cv2.contourArea(cnt)
        if area > largest_rectangle[0]:
            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]

x,y,w,h = cv2.boundingRect(largest_rectangle[1])
imgRoi=img[y:y+h,x:x+w]
plt.imshow(imgRoi,cmap = 'gray')
plt.show()

text = tess.image_to_string(imgRoi,lang='tha+digits')
print(text)