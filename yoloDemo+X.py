import cv2 as cv
from cv2 import data, resize
import numpy as np
import glob
import random
#####
import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import matplotlib.pyplot as plt
import imutils


######
 
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
    ###resize
    scale_percent = 50
    width = int(imgRoi.shape[1] * scale_percent / 100)
    height = int(imgRoi.shape[0] * scale_percent / 100)
    dsize = (width, height)
    resizeimg = cv.resize(imgRoi,dsize)
    plt.imshow(imgRoi,cmap = 'gray')
    cv.imshow('Roi',resizeimg)
    ##add kernel shapened######
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv.filter2D(resizeimg, -1, kernel)
    cv.imwrite("data/flp"+str(id)+".jpg",sharpened)

def checkPlate(id):
    # Load Yolo
    net = cv.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
    # Name custom object
    classes = ["license plate"]

    # Images path
    images_path = glob.glob(r"G:\ProjectI\Yolo\Project_yolo\data\*.jpg")

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Insert here the path of your images
    random.shuffle(images_path)
    # loop through all the images
    for img_path in images_path:
        # Loading image
        img = cv.imread(img_path)
        img = cv.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv.putText(img, label, (x, y + 30), font, 3, color, 2)
        
        cv.imwrite("data/check"+str(id)+".jpg",img)
        id = id+1

        cv.imshow("Check", img)


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
    #checkPlate(id)

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

