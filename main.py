from ultralytics import YOLO
import cv2
import os

# Load trained YOLO model
model = YOLO("best.pt")  

# Input & output folders
input_folder = "test_images"
output_folder = "results"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Severity thresholds (Adjust if needed)
MILD_THRESHOLD = 5      # Number of acne spots
MODERATE_THRESHOLD = 15 # Number of acne spots
SEVERE_AREA = 5000      # Total acne area threshold

def get_severity(num_spots, total_area):
    """Determine acne severity based on number of spots & area"""
    if num_spots <= MILD_THRESHOLD:
        return "Mild"
    elif num_spots <= MODERATE_THRESHOLD or total_area < SEVERE_AREA:
        return "Moderate"
    else:
        return "Severe"

# Process images
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)

    # Run YOLO detection
    results = model(img_path)

    for result in results:
        image = cv2.imread(img_path)
        num_acne = 0
        total_acne_area = 0

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            acne_area = (x2 - x1) * (y2 - y1)  # Calculate area
            conf = box.conf[0].item()  # Confidence score

            # Draw rectangle & label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Acne {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update severity metrics
            num_acne += 1
            total_acne_area += acne_area

        # Get severity classification
        severity = get_severity(num_acne, total_acne_area)
        print(f"Image: {img_name} | Acne Spots: {num_acne} | Total Area: {total_acne_area} | Severity: {severity}")

        # Save severity score on the image
        cv2.putText(image, f"Severity: {severity}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Save output image
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, image)

print("✅ Acne severity detection completed! Check the 'results' folder for output images.")
