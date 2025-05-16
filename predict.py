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

# ✅ Get MongoDB connection details from .env
MONGO_URI = os.getenv("MONGO_URI")  # Should be usersDB
DB_NAME = os.getenv("DB_NAME")  # Should be "usersDB"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "uploads")

# Initialize Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

try:
    client = MongoClient(MONGO_URI)
    db = client.get_database(DB_NAME)  # ✅ Ensures correct database
    collection = db[COLLECTION_NAME]  # ✅ Ensures correct collection

    print(f"✅ Connected to MongoDB: {db.name}, Collection: {collection.name}")
except Exception as e:
    print(f"❌ ERROR: Failed to connect to MongoDB: {e}")

# Load YOLO model
yolo = YOLO('BOUNDING_BOXES_YOLO.pt')

# Load ViT model from timm
vit_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
vit_model.load_state_dict(torch.load('vit_pothole_crack_model(1).pt', map_location=torch.device('cpu')))
vit_model.eval()

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
            'high': "Remove the damaged area and replace it with concrete filler or a similar material."
        },
        'pothole': {
            'low': "Use cold mix asphalt for temporary repairs.",
            'medium': "Fill the pothole with hot mix asphalt, compact it well.",
            'high': "Remove the damaged section and replace with fresh asphalt, ensuring proper compaction."
        }
    }

    actions = {
        'low': "Monitor the area periodically!.",
        'medium': "Schedule repair soon to prevent further damage.",
        'high': "Immediate repair needed to ensure road safety."
    }

    solution = solutions[label][severity]
    action = actions[severity]
    return action, solution

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve image
        file = request.files.get('image')
        if not file:
            return jsonify({'message': 'No image part named "image"'}), 400

        # Retrieve report ID to fetch location
        report_id = request.form.get('reportId')
        if report_id:
            try:
                doc = collection.find_one({'_id': ObjectId(report_id)})
                road_location = doc.get('roadLocation', {})
            except Exception:
                road_location = {'address': 'Unknown', 'district': 'Unknown', 'pincode': 'Unknown'}
        else:
            road_location = {'address': 'Unknown', 'district': 'Unknown', 'pincode': 'Unknown'}

        # Decode image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_h, img_w = image.shape[:2]

        labels = []
        boxes_list = []
        total_length = 0

        # Run YOLO
        results = yolo(image, conf=0.25, iou=0.5)[0]

        if results.boxes:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_length = x2 - x1
                total_length += box_length
                boxes_list.append([x1, y1, x2, y2])

                patch = image[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                patch_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                label = classify_patch(patch_pil)
                labels.append(label)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            patch_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            label = classify_patch(patch_pil)
            labels.append(label)

        # Determine severity
        length_ratio = total_length / img_w
        if length_ratio < 0.05:
            severity = "low"
        elif length_ratio < 0.2:
            severity = "medium"
        else:
            severity = "high"

        # Get action & solution
        action, solution = get_repair_solution(labels[0], severity) if labels else ("No damage detected", "No solution available")

        # Encode annotated image
        _, buffer = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'labels': labels,
            'boxes': boxes_list,
            'severity': severity,
            'action': action,
            'solution': solution,
            'location': road_location,
            'image': base64_image
        })
    except Exception as e:
        print("Error during prediction:", str(e))
        traceback.print_exc()
        return jsonify({'message': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)