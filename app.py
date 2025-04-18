from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid
import random

from treatments import TREATMENTS

# Initialize FastAPI
app = FastAPI()

#  CORS fix
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# processed images from the server
app.mount("/processed", StaticFiles(directory="results"), name="processed")

# Load trained YOLO model
model = YOLO("best.pt")  

# Create folders
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Severity thresholds
MILD_THRESHOLD = 5
MODERATE_THRESHOLD = 15
SEVERE_AREA = 5000

def analyze_acne(image_path, user_id):
    results = model(image_path)
    image = cv2.imread(image_path)
    num_acne = 0
    num_pimples = 0
    total_area = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            total_area += area

            if class_id == 0:
                num_acne += 1
                color = (0, 255, 0)
                label = "Acne"
            elif class_id == 1:
                num_pimples += 1
                color = (0, 0, 255)
                label = "Pimple"
            else:
                continue

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if num_acne < MILD_THRESHOLD:
        severity = "Mild"
    elif num_acne < MODERATE_THRESHOLD:
        severity = "Moderate"
    elif total_area > SEVERE_AREA:
        severity = "Severe"
    else:
        severity = "Moderate"

    output_filename = f"{user_id}_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(RESULTS_FOLDER, output_filename)
    cv2.imwrite(output_path, image)

    return num_acne, num_pimples, total_area, severity, output_filename

@app.post("/analyze_acne/")
async def analyze_acne_api(
    name: str = Form(...),
    age: int = Form(...),
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    with open(image_path, "wb") as buffer:
        buffer.write(file.file.read())

    num_acne, num_pimples, total_area, severity, output_image_filename = analyze_acne(image_path, user_id)
    processed_image_url = f"http://127.0.0.1:8000/processed/{output_image_filename}"
    recommended_treatments = random.sample(TREATMENTS[severity], min(3, len(TREATMENTS[severity])))

    os.remove(image_path)

    return JSONResponse(content={
        "name": name,
        "age": age,
        "user_id": user_id,
        "acne_spots": num_acne,
        "pimple_spots": num_pimples,
        "total_acne_area": total_area,
        "severity": severity,
        "processed_image": processed_image_url,
        "treatments": recommended_treatments
    })
