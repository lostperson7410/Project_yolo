import cv2 as cv
from cv2 import data
import numpy as np

#####
import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import matplotlib.pyplot as plt
import imutils


####
 
cap = cv.VideoCapture("car.mp4")
whT = 320
confThreshold =0.5
nmsThreshold= 0.2

id = 0


#### LOAD MODEL
## Coco Names
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
## Model Files
modelConfiguration = "yolov3-tiny.cfg"
modelWeights = "yolov3-tiny.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
 
def findObjects(outputs,img,id):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                #print(classIds) #2car 3motor
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    
    if classIds == [2]:
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            #print(x,y,w,h)
            cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
            cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            if classIds == [2]:
                crop = img[y:y+h,x:x+w]

                ####Wirtecrop###
                cv.imwrite("data/pic"+str(id)+".jpg",crop)
                id = id+1
                cv.imshow('crop',crop)
            
            crop_img = img[y:y+h,x:x+w]
            cv.imshow('crop',crop_img)
            findlicenesPlates(crop_img)
            
def findlicenesPlates(img):
    
    ##lsp##

    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(imgGray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    contours,h = cv.findContours(thresh,1,2)
    largest_rectangle = [0,0]

    for cnt in contours:
        approx = cv.approxPolyDP(cnt,0.018*cv.arcLength(cnt,True),True)
        if len(approx)==4:
            area = cv.contourArea(cnt)
            if area > largest_rectangle[0]:
                largest_rectangle = [cv.contourArea(cnt), cnt, approx]

    x,y,w,h = cv.boundingRect(largest_rectangle[1])

    
    imgRoi=img[y:y+h,x:x+w]
    plt.imshow(imgRoi,cmap = 'gray')
    cv.imshow('Shapened',imgRoi)
    cv.imwrite("data/pic"+str(id)+".jpg",imgRoi)


while True:
    success, img = cap.read()
 
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img,id)
    id = id+1
 
    cv.imshow('Image', img)
    cv.waitKey(1)

####################XcodeX#################

 ##lsp##
#           imgGray = cv.cvtColor(crop,cv.COLOR_BGR2GRAY)
#            thresh = cv.adaptiveThreshold(imgGray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
#            contours,h = cv.findContours(thresh,1,2)
#            largest_rectangle = [0,0]
#            for cnt in contours:
#               approx = cv.approxPolyDP(cnt,0.018*cv.arcLength(cnt,True),True)
#                if len(approx)==4:
#                    area = cv.contourArea(cnt)
#                  if area > largest_rectangle[0]:
#                       largest_rectangle = [cv.contourArea(cnt), cnt, approx]
#            x,y,w,h = cv.boundingRect(largest_rectangle[1])
#            imgRoi=img[y:y+h,x:x+w]
#           plt.imshow(imgRoi,cmap = 'gray')
#            cv.imshow('Shapened',imgRoi)

