import cv2
import numpy as np
import joblib

# Fungsi untuk ekstraksi fitur dari gambar (sama seperti sebelumnya)
def extract_features(image_path, size=(128, 128)):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, size)
    hsv_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
    
    hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    hist_saturation = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
    
    hist_hue = hist_hue.flatten()
    hist_saturation = hist_saturation.flatten()
    hist_value = hist_value.flatten()
    
    features = np.concatenate((hist_hue, hist_saturation, hist_value))
    return features

# Memuat model, label encoder, dan scaler yang telah disimpan
model = joblib.load('model_naive_bayes_cabai.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Fungsi untuk deteksi gambar baru
def detect_image(image_path):
    features = extract_features(image_path)
    features = features.reshape(1, -1)  # Membuat dimensi fitur sesuai dengan model
    
    # Normalisasi fitur menggunakan scaler yang telah disimpan
    features = scaler.transform(features)
    
    # Prediksi menggunakan model yang sudah dilatih
    prediction = model.predict(features)
    
    # Mengembalikan label yang diprediksi menggunakan label encoder
    predicted_label = label_encoder.inverse_transform(prediction)
    
    return predicted_label[0]

# Uji deteksi gambar baru
image_path = 'test_predict/sehat6.jpg'  # Ganti dengan path gambar yang akan dideteksi
result = detect_image(image_path)
print(f'Prediksi: {result}')
