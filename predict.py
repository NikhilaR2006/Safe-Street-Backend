import os
import torch
import cv2
import timm
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print(f"🔍 MONGO_URI: {os.getenv('MONGO_URI')}")
print(f"🔍 DB_NAME: {os.getenv('DB_NAME')}")
print(f"🔍 COLLECTION_NAME: {os.getenv('COLLECTION_NAME')}")

# MongoDB connection details
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "uploads")

# Initialize Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://safe-street-frontend.onrender.com", "http://localhost:3000"]}})

# Connect to MongoDB
try:
    client = MongoClient(MONGO_URI)
    db = client.get_database(DB_NAME)
    collection = db[COLLECTION_NAME]
    print(f"✅ Connected to MongoDB: {db.name}, Collection: {collection.name}")
except Exception as e:
    print(f"❌ ERROR: Failed to connect to MongoDB: {e}")

# Load YOLO model (Consider using 'yolov8n.pt' if you face memory issues)
yolo = YOLO('BOUNDING_BOXES_YOLO.pt')
# yolo = YOLO('yolov8n.pt')  # Uncomment this line to use smaller model

# Load quantized ViT model
vit_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
vit_model = torch.quantization.quantize_dynamic(
    vit_model, {torch.nn.Linear}, dtype=torch.qint8
)
vit_model.load_state_dict(torch.load('vit_pothole_crack_model_quantized.pt', map_location='cpu'))
vit_model.eval()

# ViT Classes
vit_classes = ['pothole', 'crack']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def classify_patch(patch_img):
    patch = transform(patch_img).unsqueeze(0)
    with torch.no_grad():
        output = vit_model(patch)
        _, predicted = torch.max(output, 1)
        return vit_classes[predicted.item()]

def get_repair_solution(label, severity):
    solutions = {
        'crack': {
            'low': "Use crack filler material to fill small cracks.",
            'medium': "Clean the crack area and use epoxy resin for better sealing.",
            'high': "Perform deep crack routing and filling, or consider partial resurfacing."
        },
        'pothole': {
            'low': "Apply cold patch for temporary fix.",
            'medium': "Clean pothole and apply hot mix asphalt.",
            'high': "Excavate damaged area and reconstruct the base before patching."
        }
    }
    return solutions.get(label, {}).get(severity, "No repair info available.")

# Example route
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

# Add your actual prediction route(s) below

# REQUIRED: Bind Flask app to a port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
