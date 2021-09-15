import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image
import glob

path = glob.glob("data/pic/*.jpg")


img =[cv2.imread(path) for flie in path]

#img = cv2.resize(img, (620,480) )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
gray = cv2.bilateralFilter(gray, 11, 17, 17)

edged = cv2.Canny(gray, 30, 200)

nts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(nts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                # if our approximated contour has four points, then
                # we can assume that we have found our screen
                if len(approx) == 4:
                      screenCnt = approx
                      break
                  
if screenCnt is None:
 detected = 0
 print ("No contour detected")
else:
 detected = 1

if detected == 1:
 cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)   

mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

#id=1
#cv2.imwrite("data/pic"+str(id)+".jpg",Cropped)


cv2.imshow('Real',img)
cv2.imshow('gray',gray)
cv2.imshow('edged',edged)
cv2.imshow('new_image',new_image)
cv2.imshow('Cropped',Cropped)

text = pytesseract.image_to_string(Cropped,lang='tha+digits')
print("Detected Number is:",text)

cv2.waitKey(0)
cv2.destroyAllWindows()