import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Baca Data CSV
try:
    data = pd.read_csv('data_hand.csv')
    print("Data berhasil dibaca!")
except FileNotFoundError:
    print("Error: File data_hand.csv tidak ditemukan. Rekam data dulu!")
    exit()

# Cek apakah datanya cukup
if len(data) < 10:
    print("PERINGATAN: Data kamu sangat sedikit! Akurasi mungkin jelek atau error.")
    print("Saran: Rekam minimal 30-50 data per huruf.")

# 2. Pisahkan Fitur (X) dan Label (y)
# X adalah koordinat (soal), y adalah hurufnya (kunci jawaban)
X = data.drop('label', axis=1) 
y = data['label']

# 3. Bagi Data: 80% untuk Belajar (Train), 20% untuk Ujian (Test)
# Kita pakai try-except karena kalau datanya terlalu sedikit, split bisa error
try:
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
except ValueError:
    # Jika data terlalu sedikit untuk stratify, kita split biasa saja
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 4. Inisialisasi Model (Kita pakai Random Forest - Kumpulan Pohon Keputusan)
model = RandomForestClassifier()

# 5. Mulai Melatih (Training)
print("Sedang melatih AI...")
model.fit(x_train, y_train)

# 6. Evaluasi (Tes Ujian)
y_prediction = model.predict(x_test)
score = accuracy_score(y_pred=y_prediction, y_true=y_test)

print(f"Pelatihan Selesai! Tingkat Akurasi: {score * 100:.2f}%")

# 7. Simpan Model ke File
f = open('model_sibi.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

print("Model tersimpan di file 'model_sibi.p'")