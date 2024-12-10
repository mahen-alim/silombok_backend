import os
import io
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)

# CORS: Allow requests from a specific origin (change to your frontend URL)
CORS(app, resources={r"/*": {"origins": "*"}})

# Paths for model, label encoder, and scaler files
model_path = 'model_naive_bayes_cabai.pkl'
label_encoder_path = 'label_encoder.pkl'
scaler_path = 'scaler.pkl'

# Check if the model, label encoder, and scaler exist
if not os.path.exists(model_path) or not os.path.exists(label_encoder_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model, Label Encoder, atau Scaler tidak ditemukan. Pastikan file ada di direktori yang sesuai.")

# Load the pre-trained model, label encoder, and scaler
model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)
scaler = joblib.load(scaler_path)

# Function to extract features from the image
def extract_features(image, size=(128, 128)):
    """Extract HSV features from the image, including resizing."""
    try:
        # Resize the image to the specified size
        image_resized = cv2.resize(image, size)
        
        # Convert the image from BGR (OpenCV) to HSV
        hsv_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
        
        # Compute histograms for each channel (Hue, Saturation, Value)
        hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        hist_saturation = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        hist_value = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        
        # Flatten the histograms into a single feature vector
        features = np.concatenate((hist_hue.flatten(), hist_saturation.flatten(), hist_value.flatten()))
        return features
    except Exception as e:
        raise ValueError(f"Error extracting features from image: {str(e)}")

# Function to convert PIL image to OpenCV (numpy array) format
def pil_to_opencv_image(pil_image):
    """Convert PIL image to OpenCV image (numpy array)."""
    try:
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1]  # Convert RGB to BGR
        return open_cv_image
    except Exception as e:
        raise ValueError(f"Error converting PIL image to OpenCV: {str(e)}")

# Function to detect disease (healthy or rotten) from the model
def detect_disease(image):
    """Detect disease in the image."""
    try:
        image_cv = pil_to_opencv_image(image)
        features = extract_features(image_cv)
        
        # Normalize the features with the pre-trained scaler
        features = scaler.transform([features])  # Scaler expects 2D array
        
        # Predict the condition of the chili (healthy or rotten)
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]  # Convert to actual label
        return predicted_label
    except Exception as e:
        raise ValueError(f"Error detecting disease: {str(e)}")

# Flask route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    # Validate image file type
    if not image_file.content_type.startswith('image/'):
        return jsonify({'error': 'File is not an image'}), 400

    try:
        # Open the image with PIL
        image = Image.open(io.BytesIO(image_file.read()))
        
        # Predict the condition of the chili (healthy or rotten)
        result = detect_disease(image)
        
        # Recommendations based on prediction
        recommendations = {
            "sehat": {
                "penyiraman": "Siram tanaman setiap pagi dan sore",
                "perawatan": "Periksa kondisi daun dan batang secara rutin",
                "perkiraan_waktu_panen": "Tanaman siap panen dalam 2-3 minggu",
                "kerusakan_fisik": "Pastikan tidak ada kerusakan fisik pada buah/tanaman"
            },
            "busuk": {
                "penyiraman": "Kurangi penyiraman untuk menghindari pembusukan lebih lanjut",
                "perawatan": "Buang bagian yang busuk dan jaga area tetap kering",
                "perkiraan_waktu_panen": "Tidak disarankan untuk dipanen",
                "kerusakan_fisik": "Periksa apakah pembusukan menyebar, segera pisahkan dari buah/tanaman sehat"
            }
        }

        # Return the prediction and recommendations in JSON format
        return jsonify({
            'prediction': result,
            'rekomendasi': recommendations.get(result, {})
        })
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)