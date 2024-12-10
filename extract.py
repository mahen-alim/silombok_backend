import pandas as pd
import numpy as np
import cv2
import os

# Kondisi kolam sehat
path_sehat = "dataset_train/sehat"
data_sehat = os.listdir(path_sehat)
NamaFile = []
AvgH = []
AvgS = []
AvgV = []
Label = []

for gbr in data_sehat:
    gbr_read = cv2.imread(os.path.join(path_sehat, gbr))
    gbr_hsv = cv2.cvtColor(gbr_read, cv2.COLOR_BGR2HSV)
    
    # Ekstraksi rata-rata HSV
    meanH = np.mean(gbr_hsv[:, :, 0])
    meanS = np.mean(gbr_hsv[:, :, 1])
    meanV = np.mean(gbr_hsv[:, :, 2])
    
    NamaFile.append(gbr)  # Tambahkan nama file
    AvgH.append(meanH)
    AvgS.append(meanS)
    AvgV.append(meanV)
    Label.append(1)  # Sehat

# Dataframe untuk kondisi sehat
data_sehat = pd.DataFrame({
    'Nama File': NamaFile,
    'Avg H': AvgH,
    'Avg S': AvgS,
    'Avg V': AvgV,
    'Label': Label
})

# Kondisi kolam busuk
path_busuk = "dataset_train/busuk"
data_busuk = os.listdir(path_busuk)
NamaFile = []
AvgH = []
AvgS = []
AvgV = []
Label = []

for gbr in data_busuk:
    gbr_read = cv2.imread(os.path.join(path_busuk, gbr))
    gbr_hsv = cv2.cvtColor(gbr_read, cv2.COLOR_BGR2HSV)
    
    # Ekstraksi rata-rata HSV
    meanH = np.mean(gbr_hsv[:, :, 0])
    meanS = np.mean(gbr_hsv[:, :, 1])
    meanV = np.mean(gbr_hsv[:, :, 2])
    
    NamaFile.append(gbr)  # Tambahkan nama file
    AvgH.append(meanH)
    AvgS.append(meanS)
    AvgV.append(meanV)
    Label.append(0)  # Busuk

# Dataframe untuk kondisi busuk
data_busuk = pd.DataFrame({
    'Nama File': NamaFile,
    'Avg H': AvgH,
    'Avg S': AvgS,
    'Avg V': AvgV,
    'Label': Label
})

# Gabungkan kedua dataframe
total = pd.concat([data_sehat, data_busuk], ignore_index=True)

# Ekspor ke file Excel
total.to_excel("Extraksi_Warna_Cabai(dataset_new).xlsx", index=False)

print("Ekstraksi fitur berhasil, hasil disimpan dalam file 'Extraksi_Warna_Cabai(dataset_new).xlsx'")