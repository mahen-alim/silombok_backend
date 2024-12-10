import os
import cv2
import numpy as np
import joblib

# Fungsi untuk ekstraksi fitur dari gambar (sama dengan yang digunakan pada pelatihan)
def extract_features(image_path, size=(128, 128)):
    """Mengambil fitur HSV dari gambar, termasuk resizing gambar"""
    image = cv2.imread(image_path)

    # Menangani jika gambar tidak dapat dibaca
    if image is None:
        print(f"[WARNING] Gagal membaca gambar: {image_path}")
        return None

    # Menyesuaikan ukuran gambar agar seragam
    image_resized = cv2.resize(image, size)

    # Mengonversi gambar dari BGR (OpenCV default) ke HSV
    hsv_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

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

# Fungsi untuk memuat model dan melakukan prediksi
def predict_image(image_path):
    # Memuat model, Label Encoder, dan Scaler
    model_filename = 'model_naive_bayes_cabai.pkl'
    label_encoder_filename = 'label_encoder.pkl'
    scaler_filename = 'scaler.pkl'

    if not os.path.exists(model_filename) or not os.path.exists(label_encoder_filename) or not os.path.exists(scaler_filename):
        print("[ERROR] File model atau pendukung tidak ditemukan.")
        return

    # Load model, label encoder, dan scaler
    model = joblib.load(model_filename)
    label_encoder = joblib.load(label_encoder_filename)
    scaler = joblib.load(scaler_filename)

    # Ekstraksi fitur dari gambar
    features = extract_features(image_path)
    if features is None:
        print("[ERROR] Gagal mengekstraksi fitur dari gambar.")
        return

    # Normalisasi fitur
    features_scaled = scaler.transform([features])

    # Prediksi label
    prediction = model.predict(features_scaled)

    # Mengembalikan label asli dari hasil prediksi
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    print(f"Hasil prediksi: {predicted_label}")
    return predicted_label

# Contoh penggunaan
image_path = 'test_predict/sehat11.jpg'  # Ganti dengan path ke gambar yang ingin diprediksi
predict_image(image_path)