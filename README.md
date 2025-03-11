# 🧑‍⚕️ Acne Detection API using YOLOv8

This project provides an API for detecting acne in face images using **YOLOv8**. It highlights acne spots and generates a severity score based on the detected acne area.

---
## 🚀 Features
✅ Detects acne spots on facial images  
✅ Generates severity score (Mild, Moderate, Severe)  
✅ Returns a processed image with acne highlighted  
✅ Built using **FastAPI** & **YOLOv8**  
✅ Simple API to receive user data and image  

---
## 🏗 Folder Structure
```
├── dataset/                  # Dataset directory
│   ├── train/
│   │   ├── images/
│   │   ├── labels/
│   ├── valid/
│   │   ├── images/
│   │   ├── labels/
│   ├── test/
│   │   ├── images/
│   │   ├── labels/
├── models/                   # Trained YOLOv8 model
├── results/                   # Processed images output
├── main.py                    # FastAPI backend
├── requirements.txt            # Dependencies
├── config.yaml                 # YOLOv8 dataset config
├── README.md                   # Project documentation
```

---
## 🔧 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/acne-detection-api.git
cd acne-detection-api
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows, use: .venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---
## 🎯 Usage
### 1️⃣ Run the API Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2️⃣ Open API Documentation (Swagger UI)
Go to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

You can test the API here by uploading an image and entering user details.

---
## 🔗 API Endpoints
### 🔹 **POST /analyze_acne/**
📌 **Request:** (multipart/form-data)
```json
{
  "name": "John Doe",
  "age": 22,
  "file": "image.jpg"
}
```
📌 **Response:** (JSON)
```json
{
  "name": "John Doe",
  "age": 22,
  "acne_spots": 10,
  "total_acne_area": 4500,
  "severity": "Moderate",
  "processed_image": "results/image.jpg"
}
```

---
## 🧪 Testing the API
### **1️⃣ Using Swagger UI**
- Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Click on `POST /analyze_acne/`
- Upload a face image and enter `name` & `age`
- Click **Execute** to get results

### **2️⃣ Using Postman**
1. Open **Postman**
2. Create a `POST` request to `http://127.0.0.1:8000/analyze_acne/`
3. In `Body` (form-data):
   - `name`: Text (e.g., "John Doe")
   - `age`: Text (e.g., "22")
   - `file`: File (upload an image)
4. Click **Send** to receive results

### **3️⃣ Using cURL (Terminal)**
```bash
curl -X 'POST' 'http://127.0.0.1:8000/analyze_acne/' \
  -F 'name=John Doe' \
  -F 'age=22' \
  -F 'file=@path/to/your/image.jpg'
```

---
## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

---
## 👨‍💻 Author
Developed by Seneth Mendis Kumarasignha  
🔗 GitHub: [github.com/senethmendis](https://github.com/senethmendis)  

