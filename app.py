from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import logging
from dotenv import load_dotenv  # ✅ Load environment variables

# ✅ Load environment variables from .env
load_dotenv()

print(f"🔍 MONGO_URI: {os.getenv('MONGO_URI')}")
print(f"🔍 DB_NAME: {os.getenv('DB_NAME')}")
print(f"🔍 COLLECTION_NAME: {os.getenv('COLLECTION_NAME')}")

# ✅ Get MongoDB connection details from .env
MONGO_URI = os.getenv("MONGO_URI")  # Should be usersDB
DB_NAME = os.getenv("DB_NAME")  # Should be "usersDB"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "uploads")

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # ✅ Allow frontend

# ✅ Connect to MongoDB
try:
    client = MongoClient(MONGO_URI)
    db = client.get_database(DB_NAME)  # ✅ Ensures correct database
    collection = db[COLLECTION_NAME]  # ✅ Ensures correct collection

    print(f"✅ Connected to MongoDB: {db.name}, Collection: {collection.name}")
except Exception as e:
    print(f"❌ ERROR: Failed to connect to MongoDB: {e}")


# ✅ Define Model Path
MODEL_PATH = "binary_road_classif_fixed.pth"
8
# ✅ Check if Model File Exists
if not os.path.exists(MODEL_PATH):
    logging.error(f"❌ ERROR: Model file '{MODEL_PATH}' not found! Please check your backend directory.")
    exit(1)

# ✅ Check Device (CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"🚀 Using device: {device}")

# ✅ Define CNN Model
class RoadClassifierCNN(nn.Module):
    def __init__(self, in_features):
        super(RoadClassifierCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# ✅ Detect `in_features`
def get_in_features():
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 128, 128)
        conv_out = RoadClassifierCNN(1).conv_layers(dummy_input)
        return conv_out.view(1, -1).size(1)

# ✅ Load Model
try:
    in_features = get_in_features()
    model = RoadClassifierCNN(in_features).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ ERROR loading model: {e}")
    exit(1)

# ✅ Define Image Preprocessing
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ Health Check API
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"message": "Flask server is running!", "status": "OK"}), 200

# ✅ Store Image API (Saves to MongoDB)
@app.route('/api/store', methods=['POST'])
def store_image():
    try:
        data = request.json
        if not data:
            logging.error("❌ No data received!")
            return jsonify({"error": "No data received"}), 400

        logging.info(f"📥 Received Data: {data}")

        result = collection.insert_one(data)
        logging.info(f"✅ MongoDB Inserted ID: {result.inserted_id}")

        return jsonify({"success": True, "message": "Image stored successfully", "id": str(result.inserted_id)}), 200

    except Exception as e:
        logging.error(f"❌ ERROR in /api/store: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Fetch Stored Images
@app.route('/api/images', methods=['GET'])
def get_images():
    try:
        images = list(collection.find({}, {"_id": 0}))
        logging.info(f"📂 Retrieved {len(images)} Images from MongoDB")
        return jsonify({"success": True, "images": images}), 200

    except Exception as e:
        logging.error(f"❌ ERROR in /api/images: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Fetch User's Uploads
@app.route('/api/uploads', methods=['GET'])
def get_user_uploads():
    try:
        user_email = request.args.get('email')
        if not user_email:
            return jsonify({"error": "Email parameter is required"}), 400

        # Find all uploads for the specific user
        uploads = list(collection.find({"userEmail": user_email}, {"_id": 0}))
        
        # Sort by upload date in descending order (newest first)
        uploads.sort(key=lambda x: x.get('uploadedAt', ''), reverse=True)
        
        logging.info(f"📂 Retrieved {len(uploads)} uploads for user: {user_email}")
        return jsonify({"success": True, "uploads": uploads}), 200

    except Exception as e:
        logging.error(f"❌ ERROR in /api/uploads: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            logging.error("❌ No image provided!")
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        filename = file.filename
        logging.info(f"📂 Received file: {filename}")

        image = Image.open(file).convert('RGB')
        image = test_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        class_names = ["Not Road", "Road"]
        classification = class_names[predicted.item()]
        logging.info(f"✅ Classified as {classification} with {confidence.item():.4f} confidence")
        return jsonify({
            'class': classification,
            'confidence': round(confidence.item(), 4),
            'filename': filename,
            'imageUrl': f"http://localhost:5001/uploads/{filename}"
        })

    except Exception as e:
        logging.error(f"❌ ERROR in /predict: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Start Flask Server
if __name__ == '__main__':
    logging.info("🚀 Starting Flask server on http://127.0.0.1:5001 ...")
    app.run(host="0.0.0.0", port=5001, debug=True)
