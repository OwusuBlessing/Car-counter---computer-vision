# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 01:22:45 2023

@author: User
"""

import numpy as np
import ultralytics
from ultralytics import YOLO
import cv2

#Running YOLO basically
weight_path = "C:/Users/User/Desktop/Blessing_AI/Object_detection_yolo/My_learning/yolov8n.pt"
model = YOLO(weight_path)
path = "C:/Users/User/Desktop/Blessing_AI/Object_detection_yolo/Chapter 5 - Running Yolo/Images/3.png"
result = model(path,show=True)
cv2.waitKey(0)