from ultralytics import YOLO
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the YOLOv8 model (pre-trained)
model = YOLO("yolov8n.pt")  # You can replace with a fine-tuned model later

# Check if CUDA is available (use GPU if possible)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
