from ultralytics import YOLO
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the YOLOv8 model 
model = YOLO("yolov8n.pt") 

# Check if CUDA is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
