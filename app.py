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

# Define possible treatments for each severity level
TREATMENTS = {
    "Mild": [
        "Use a gentle cleanser with salicylic acid",
        "Apply a non-comedogenic moisturizer",
        "Avoid touching your face frequently",
        "Use sunscreen with SPF 30+",
    ],
    "Moderate": [
        "Use a benzoyl peroxide or salicylic acid cleanser",
        "Apply topical retinoids",
        "Consider a mild chemical exfoliant (AHA/BHA)",
        "Hydrate with an oil-free moisturizer",
    ],
    "Severe": [
        "Consult a dermatologist for prescription treatments",
        "Consider oral antibiotics or isotretinoin",
        "Use medical-grade acne treatments",
        "Follow a strict skincare routine with dermatologist-approved products",
    ],
}

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["http://localhost:3000"] to restrict it to your frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Serve processed images for easy access
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
    """ Runs YOLO on an image & calculates severity score """
    results = model(image_path)
    image = cv2.imread(image_path)
    num_acne = 0
    total_area = 0

    for result in results:
        for box in result.boxes:
            num_acne += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            total_area += area
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, "Acne", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Determine severity
    if num_acne < MILD_THRESHOLD:
        severity = "Mild"
    elif num_acne < MODERATE_THRESHOLD:
        severity = "Moderate"
    elif total_area > SEVERE_AREA:
        severity = "Severe"
    else:
        severity = "Moderate"

    # Save processed image with a unique filename
    output_filename = f"{user_id}_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(RESULTS_FOLDER, output_filename)
    cv2.imwrite(output_path, image)

    return num_acne, total_area, severity, output_filename

@app.post("/analyze_acne/")
async def analyze_acne_api(
    name: str = Form(...),
    age: int = Form(...),
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    """ API endpoint to receive user data & image, and return acne analysis """

    # Save uploaded image with a unique filename
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    with open(image_path, "wb") as buffer:
        buffer.write(file.file.read())

    # Run acne analysis
    num_acne, total_area, severity, output_image_filename = analyze_acne(image_path, user_id)

    # Construct full URL for processed image
    processed_image_url = f"http://127.0.0.1:8000/processed/{output_image_filename}"

    # Select 2-3 random treatments based on severity
    recommended_treatments = random.sample(TREATMENTS[severity], min(3, len(TREATMENTS[severity])))

    # Clean up temp uploaded image
    os.remove(image_path)

    return JSONResponse(content={
        "name": name,
        "age": age,
        "user_id": user_id,
        "acne_spots": num_acne,
        "total_acne_area": total_area,
        "severity": severity,
        "processed_image": processed_image_url,
        "treatments": recommended_treatments
    })
