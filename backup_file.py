from ultralytics import YOLO
import cv2
import os

# Load your trained YOLO model
model = YOLO("best.pt")  # Change path if necessary

# Folder containing images for detection
input_folder = "test_images"
output_folder = "results"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through images in the input folder
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)

    # Run YOLO model on the image
    results = model(img_path)

    # Draw bounding boxes on the image
    for result in results:
        image = cv2.imread(img_path)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score

            # Draw rectangle around acne
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Acne {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the processed image
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, image)

print("✅ Acne detection completed! Check the 'results' folder for output images.")

def calculate_severity(results):
    total_acne = len(results[0].boxes)
    severity = "Mild" if total_acne < 5 else "Moderate" if total_acne < 15 else "Severe"
    return total_acne, severity

# Example usage
results = model("test_image.jpg")
total_acne, severity = calculate_severity(results)
print(f"Total Acne: {total_acne}, Severity: {severity}")

