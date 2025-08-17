# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# ------------------ Model Definition ------------------
IMG_SIZE = 128

class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (IMG_SIZE//4) * (IMG_SIZE//4), 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
    'Tomato_healthy'
]

# ------------------ Flask App ------------------
app = Flask(__name__)
# List of allowed frontend origins
origins = [
    "https://vercel-frontend-hazel-nu.vercel.app", 
    "http://localhost:3000"
]

CORS(app, origins=origins, methods=["GET", "POST"])

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device: CPU on Render, MPS for mac locally
device = torch.device("cpu")
model = PlantCNN(num_classes=len(class_names))

# Load model weights
model_path = "crop_disease_model.pth"  # make sure it's in same folder as app.py
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ------------------ Image Transform ------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ------------------ API Endpoint ------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Load image and preprocess
    image = Image.open(file_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]

    result = {
        "filename": file.filename,
        "prediction": label,
        "message": f"The model predicts: {label}"
    }

    return jsonify(result)

# ------------------ Run App ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
