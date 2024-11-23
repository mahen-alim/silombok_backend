import os
import cv2
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib  # Untuk menyimpan model

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

# Path folder dataset
dataset_dir = 'dataset_cabai'

# Daftar untuk fitur dan label
features = []
labels = []

# Menelusuri folder "busuk" dan "sehat"
for label in ['sehat', 'busuk']:
    folder_path = os.path.join(dataset_dir, label)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            file_path = os.path.join(folder_path, filename)
            hist = extract_features(file_path)
            features.append(hist)
            labels.append(label)

# Mengonversi daftar fitur dan label menjadi array numpy
features = np.array(features)
labels = np.array(labels)

# Mengonversi label menjadi angka
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Melatih model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Prediksi menggunakan data uji
y_pred = model.predict(X_test)

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
joblib.dump(model, model_filename)
print(f'Model berhasil disimpan ke dalam file: {model_filename}')

# Menyimpan Label Encoder untuk digunakan nanti
label_encoder_filename = 'label_encoder.pkl'
joblib.dump(label_encoder, label_encoder_filename)
print(f'Label Encoder berhasil disimpan ke dalam file: {label_encoder_filename}')
