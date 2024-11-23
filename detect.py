import os
import io
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.preprocessing import LabelEncoder

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app)

# Memuat model dan LabelEncoder yang telah disimpan
model = joblib.load('model_naive_bayes_cabai.pkl')  # Model Naive Bayes yang telah dilatih
label_encoder = joblib.load('label_encoder.pkl')  # Label encoder yang digunakan untuk label encoding

# Fungsi untuk ekstraksi fitur dari gambar
def extract_features(image_path):
    # Membaca gambar menggunakan OpenCV
    image = cv2.imread(image_path)
    
    # Mengonversi gambar dari BGR (default OpenCV) ke HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Menghitung histogram untuk setiap saluran (Hue, Saturation, Value)
    hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])  # Histogram Hue
    hist_saturation = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])  # Histogram Saturation
    hist_value = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])  # Histogram Value
    
    # Flattening the histograms to create a single feature vector
    hist_hue = hist_hue.flatten()
    hist_saturation = hist_saturation.flatten()
    hist_value = hist_value.flatten()
    
    # Menggabungkan histogram menjadi satu vektor fitur
    features = np.concatenate((hist_hue, hist_saturation, hist_value))
    
    return features

# Fungsi untuk mengekstrak fitur dari gambar yang diunggah melalui Flask
def extract_hsv_features_from_image(image):
    # Mengonversi gambar dari PIL ke numpy array (BGR)
    image = np.array(image)
    
    # Menghitung fitur HSV menggunakan fungsi yang sama dengan pelatihan
    return extract_features(image)

# Fungsi untuk deteksi penyakit cabai (sehat atau busuk)
def detect_disease(image):
    # Ekstrak fitur HSV dari gambar
    hsv_features = extract_hsv_features_from_image(image)
    
    # Menstandarisasi data fitur dengan scaler yang digunakan pada pelatihan (jika ada)
    # Jika menggunakan scaler, tambahkan kode scaler di sini sebelum prediksi:
    # hsv_features_scaled = scaler.transform([hsv_features])  # Uncomment jika scaler digunakan
    
    # Melakukan prediksi menggunakan model Naive Bayes
    prediction = model.predict([hsv_features])
    
    # Mengembalikan hasil prediksi dalam label string
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    return predicted_label

# Rute Flask untuk menerima gambar dan memberikan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Mendapatkan file gambar yang diunggah
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))  # Membuka gambar dengan PIL
    
    # Melakukan prediksi kondisi cabai
    result = detect_disease(image)
    
    # Menambahkan rekomendasi berdasarkan hasil prediksi
    if result == "sehat":  # Jika prediksi Sehat
        rekomendasi = {
            "penyiraman": "Siram tanaman setiap pagi dan sore",
            "perawatan": "Periksa kondisi daun dan batang secara rutin",
            "perkiraan_waktu_panen": "Tanaman siap panen dalam 2-3 minggu",
            "kerusakan_fisik": "Pastikan tidak ada kerusakan fisik pada buah/tanaman"
        }
    else:  # Jika prediksi Busuk
        rekomendasi = {
            "penyiraman": "Kurangi penyiraman untuk menghindari pembusukan lebih lanjut",
            "perawatan": "Buang bagian yang busuk dan jaga area tetap kering",
            "perkiraan_waktu_panen": "Tidak disarankan untuk dipanen",
            "kerusakan_fisik": "Periksa apakah pembusukan menyebar, segera pisahkan dari buah/tanaman sehat"
        }

    # Mengirimkan hasil prediksi dan rekomendasi dalam format JSON
    return jsonify({'prediction': result, 'rekomendasi': rekomendasi})

if __name__ == '__main__':
    app.run(debug=True)

