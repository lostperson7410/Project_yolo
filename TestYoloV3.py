import cv2 as cv
import numpy as np
cap = cv.VideoCapture('car.mp4')
whT = 320
confiThreshold = 0.5
nmsThreshold= 0.2

classesFile= 'coco.names'
classes =[]
with open(classesFile,'rt')as f :
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)
#print(len(classNames))

modelConfig = 'yolov3-tiny.cfg'
modelWeight = 'yolov3-tiny.weights'

net = cv.dnn.readNetFromDarknet(modelConfig,modelWeight)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)



def findObject(outputs,img):
    hT,wT,cT =img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            score =det [5:]
            classId = np.argmax(score)
            confidence = score[classId]
            if confidence > confiThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                #print(classIds) #2car 3motor
                confs.append(float(confidence))
    print(len(bbox))
    
    indices = cv.dnn.NMSBoxes(bbox, confs, confiThreshold, nmsThreshold)
    
    if classIds == [2]:
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            #print(x,y,w,h)
            cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
            cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
            crop_img = img[y:y+h,x:x+w]
            cv.imshow('crop',crop_img)






while True: 
    success,img = cap.read()

    blob = cv.dnn.blobFromImage(img,1/255,(whT,whT),(0,0,0,),1,crop = False)
    net.setInput(blob)
    
    layerNames = net.getLayerNames()
    #print(layerNames)
    #print(net.getUnconnectedOutLayers())
    
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    outputs= net.forward(outputNames)
    #print(len(outputs))

    findObject(outputs,img)


    cv.imshow('Image',img)
    cv.waitKey(1)

