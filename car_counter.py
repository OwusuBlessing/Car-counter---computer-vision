# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 21:20:40 2023

@author: User
"""


import numpy as np
import ultralytics
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import * 

video_path = r"C:/Users/User/Desktop/Blessing_AI/Object_detection_yolo/Videos/cars.mp4"
cap = cv2.VideoCapture(video_path)
#cap.set(3, 640)
#cap.set(4,480)
#load yolo weight

weight_path = "My_learning/yolov8n.pt"
model = YOLO(weight_path)

ClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#import mask
mask_path = r"C:/Users/User/Desktop/Blessing_AI/Object_detection_yolo/My_learning/Car_counter/mask.png"
mask =  cv2.imread(mask_path)

graphics_path = r"C:/Users/User/Desktop/Blessing_AI/Object_detection_yolo/My_learning/Car_counter/graphics.png"
#tracker
tracker = Sort(max_age=20,min_hits=2,iou_threshold=0.3 )

#create line
limits = [394,297,673,297]
total_count = []
while True:
    success, img = cap.read(  )
    imageReg = cv2.bitwise_and(img, mask)
    
    #image graphis
    
    qgraphics_path = r"C:/Users/User/Desktop/Blessing_AI/Object_detection_yolo/My_learning/Car_counter/graphics.png"

    graphics = cv2.imread(graphics_path,cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,graphics,(0,0))
    results = model(imageReg,stream=True)
    detections = np.empty((0,5))
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),3)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            #x1,y1,w,h = box.xywh[0]
            x1, y1, x2, y2  = int(x1), int(y1), int(x2), int(y2) 
            w , h = x2 - x1,y2 - y1
            bbox = int(x1),int(y1),int(w),int(h)
            print(x1,y1,x2,y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
           
            #confidence
            conf = math.ceil(box.conf[0] * 100)/100
            print(conf)
            cvzone.putTextRect(img,f'{conf}',(max(0,w // 2),max(20,y1)),scale=1,thickness=1)
            
            #class name 
            cls_= int(box.cls[0])
            current_class = ClassNames[cls_]
            
            if current_class == "car"  or current_class == "truck" or current_class == "bus" or current_class == "motorbike" and conf > 0.3:
                cvzone.putTextRect(img,f'{ClassNames[cls_]}',(max(0,x1),max(20,y1)),scale = 1,thickness = 1)
                print(cls_)
                current_arry = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,current_arry))
    tracker_results = tracker.update(detections)
    for result in tracker_results:
        x1,y1,x2,y2,id_value = result
        x1, y1, x2, y2  = int(x1), int(y1), int(x2), int(y2) 
        w , h = x2 - x1,y2 - y1
        bbox = int(x1),int(y1),int(w),int(h)
        cvzone.cornerRect(img, bbox,colorR=(255,0,0),rt=1)
        cvzone.putTextRect(img,f'{int(id_value)}',(max(0,x1),max(20,y1)),scale = 1,thickness = 1)
        print(result)
        
        cx,cy = x1 + w//2,y1 + h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        
        if limits[0]<= cx <= limits[2] and limits[1] - 20 <= cy <= limits[3] + 20:
            if total_count.count(id_value) == 0:
                 total_count.append(id_value)
                 cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(255,0,0),5)
            
            
    
    cv2.putText(img,f'{len(total_count)}',(255,100),cv2.FONT_HERSHEY_COMPLEX,5,(50,50,255),8 )
    cv2.imshow("Car counter AI", img)
    cv2.waitKey(1 )
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

