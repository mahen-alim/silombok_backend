import os
import cv2
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import joblib  # Untuk menyimpan model

# Fungsi untuk ekstraksi fitur dari gambar (Menggunakan ukuran lebih besar)
def extract_features(image_path, size=(256, 256)):  # Ukuran gambar yang lebih besar
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

# Fungsi untuk melatih model dengan GridSearchCV untuk optimasi
def train_model(dataset_dir):
    features = []
    labels = []
    
    # Menelusuri folder "busuk" dan "sehat"
    for label in ['sehat', 'busuk']:
        folder_path = os.path.join(dataset_dir, label)
        
        # Memeriksa apakah folder ada
        if not os.path.exists(folder_path):
            print(f"[ERROR] Folder untuk label '{label}' tidak ditemukan.")
            continue
        
        print(f"Menelusuri folder {folder_path} untuk label {label}...")
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                file_path = os.path.join(folder_path, filename)
                hist = extract_features(file_path)
                if hist is not None:  # Hanya tambahkan gambar yang valid
                    features.append(hist)
                    labels.append(label)
    
    # Mengonversi daftar fitur dan label menjadi array numpy
    if len(features) == 0:
        print("[ERROR] Tidak ada data fitur yang valid, pelatihan gagal.")
        return
    
    features = np.array(features)
    labels = np.array(labels)

    # Mengonversi label menjadi angka
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Normalisasi fitur dengan MinMaxScaler (coba scaler lain)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # Membagi dataset menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Melatih model Naive Bayes dengan GridSearchCV untuk tuning parameter
    model = GaussianNB()
    
    # Tuning Naive Bayes menggunakan GridSearchCV
    params = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}  # Uji smoothing yang berbeda
    grid_search = GridSearchCV(model, params, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Prediksi menggunakan data uji
    y_pred = best_model.predict(X_test)

    # Mengukur akurasi model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Akurasi Model Naive Bayes: {accuracy * 100:.2f}%')

    # Menampilkan Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Menampilkan Classification Report
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("Classification Report:")
    print(class_report)

    # Menyimpan model ke dalam file menggunakan joblib
    model_filename = 'model_naive_bayes_cabai.pkl'
    joblib.dump(best_model, model_filename)
    print(f'Model berhasil disimpan ke dalam file: {model_filename}')

    # Menyimpan Label Encoder dan Scaler
    label_encoder_filename = 'label_encoder.pkl'
    joblib.dump(label_encoder, label_encoder_filename)
    print(f'Label Encoder berhasil disimpan ke dalam file: {label_encoder_filename}')
    
    scaler_filename = 'scaler.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f'Scaler berhasil disimpan ke dalam file: {scaler_filename}')

# Menentukan folder dataset
dataset_dir = 'dataset_train'

# Melatih model
train_model(dataset_dir) 