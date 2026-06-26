# Sistem Analisis Data dan Deep Learning Terintegrasi

Aplikasi desktop Tkinter yang menggabungkan **Project 1 (Data Preparation & Machine Learning)**
dan **Project 2 (Time Series Forecasting dengan LSTM)** dalam satu aplikasi.

## Instalasi

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
python main.py
```

(Tkinter biasanya sudah include di Python. Jika belum: `sudo apt-get install python3-tk` di Linux,
atau install ulang Python dari python.org di Windows/Mac dengan opsi tcl/tk dicentang.)

## Struktur File

- `main.py` — entry point aplikasi (sidebar, notebook, status bar)
- `project1_tab.py` — semua fitur Project 1 (9 sub-tab)
- `project2_tab.py` — semua fitur Project 2 (8 sub-tab)
- `lstm_model.py` — implementasi LSTM **dari nol dengan NumPy** (forward LSTM 4-gate penuh,
  backpropagation through time, optimizer Adam)
- `data_generators.py` — generator dataset sintetis
- `export_utils.py` — export CSV/Excel/PDF

## Catatan penting tentang LSTM

Karena lingkungan pembuatan aplikasi ini tidak memiliki akses internet untuk
`pip install tensorflow`, LSTM diimplementasikan dari nol di `lstm_model.py`
menggunakan NumPy murni (forward pass 4 gate: forget, input, candidate, output,
plus backpropagation through time dan optimizer Adam). Ini sudah teruji
menurunkan loss secara nyata.

**Jika di komputer Anda TensorFlow/Keras tersedia** dan Anda ingin versi
Keras (lebih cepat, lebih mudah dijelaskan di laporan sebagai "menggunakan
Keras LSTM layer"), ganti isi `lstm_model.py` dengan implementasi Keras
standar — API (`fit`, `predict`, `.history`) yang dipakai `project2_tab.py`
hanya butuh method `fit(X, y, X_val, y_val, epochs, batch_size, lr,
progress_callback, stop_flag)` dan `predict(X)`. Tidak ada perubahan lain
yang dibutuhkan di GUI.

## Tentang Dataset

Karena lingkungan ini tidak memiliki akses internet ke Kaggle, aplikasi
menyediakan tombol **"Generate Dataset Sintetis"** yang membuat data dengan
struktur serupa:

- Project 1: mirip US Accidents (>20 kolom, missing value, outlier, fitur
  kategorikal, butuh transformasi & encoding)
- Project 2: mirip Air Pollution in China 2015-2025 (time series harian
  multi-kota dengan tren, musiman, noise)

Untuk laporan/demo final, **download dataset asli** dari:
- https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
- https://www.kaggle.com/datasets/khushikyad001/air-pollution-in-china-2015-2025

lalu klik tombol **"Load CSV..."** di masing-masing tab — semua fitur
(viewer, EDA, preprocessing, ML, LSTM, evaluasi, export) bekerja sama persis
dengan dataset asli tanpa perlu ubah kode.

## Update Terbaru

### 1. Mode "Bangun dari Year + Month" (Project 2)

Jika dataset time series Anda **tidak memiliki kolom tanggal lengkap** (hanya
ada `Year`, `Month`, `Day of Week`, `Hour` — seperti yang umum terjadi pada
beberapa versi dataset polusi udara di Kaggle), aplikasi akan **otomatis
mendeteksi** hal ini saat dataset dimuat dan beralih ke mode:

> **"Bangun dari Year + Month (Agregasi Bulanan)"**

Pada mode ini, data dirata-ratakan per `Kota + Tahun + Bulan` sehingga
urutan waktunya **valid secara kronologis** (granularitas bulanan, bukan
harian). Ini lebih jujur secara metodologis dibanding memaksa membuat
urutan harian palsu dari data yang sebenarnya tidak berurutan.

Anda tetap bisa memilih mode "Kolom Tanggal Asli" secara manual di
dropdown "Sumber Sumbu Waktu" pada sub-tab Dataset Viewer Project 2, jika
dataset Anda memang punya kolom tanggal harian yang valid.

### 2. Export Laporan Lengkap (1 file, multi-sheet)

Tombol baru di **sidebar kiri** aplikasi (bagian "LAPORAN GABUNGAN"):

- **"Export Laporan Lengkap (1 file Excel, multi-sheet)"** — menggabungkan
  SEMUA hasil dari Project 1 dan Project 2 ke dalam **satu file `.xlsx`**
  dengan sheet terpisah dan terformat (header berwarna, lebar kolom
  otomatis, freeze header):
  - `Ringkasan` — info dataset, parameter, waktu ekspor
  - `P1_Eksperimen_ML` — tabel semua eksperimen RF/LR yang dijalankan
  - `P1_Statistik_Performa` — rata-rata/min/maks per model
  - `P1_Missing_Value`, `P1_Outlier_Analysis`
  - `P2_Evaluasi_LSTM` — tabel MAE/MSE/RMSE/MAPE/R²
  - `P2_Riwayat_Loss` — loss training & validation per epoch
  - `P2_Prediksi_vs_Aktual` — data aktual vs prediksi

- **"Export Laporan Lengkap (1 file PDF)"** — versi siap cetak/lampiran:
  cover, tabel-tabel di atas, plus grafik Loss dan Prediksi vs Aktual,
  semua dalam satu PDF.

Ini jauh lebih praktis untuk laporan dibanding export terpisah per tab —
tinggal lampirkan satu file ini sebagai bukti hasil eksperimen lengkap.

## Fitur yang Sudah Diimplementasikan

### Project 1
1. Dataset Viewer (info, tipe data, preview)
2. Missing Value Analysis (tabel + grafik bar)
3. Outlier Analysis (IQR + boxplot)
4. Data Encoding (Label / One-Hot)
5. Data Transformation (Standard/MinMax/RobustScaler)
6. Feature Engineering (Feature Construction, Feature Selection SelectKBest, PCA)
7. Eksperimen ML (Random Forest & Logistic Regression, tabel hasil akumulatif)
8. Statistik Performa (rata-rata, minimum, maksimum per model)
9. Visualisasi (Histogram, Boxplot, Correlation Heatmap, Feature Importance,
   Confusion Matrix, Perbandingan Accuracy)
10. Export CSV / Excel / PDF dari tabel hasil

### Project 2
1. Dataset Viewer + pemilihan kolom tanggal/target/filter (mis. per kota)
2. EDA (statistik deskriptif, missing, outlier, Line Chart, Histogram, Boxplot,
   Trend Analysis, Seasonal Analysis, Rolling Mean, Rolling Std, Heatmap Korelasi)
3. Preprocessing (Missing Value Handling, Scaling, Windowing/Sequence, Train-Test Split)
4. Deep Learning LSTM dengan parameter Epoch, Batch Size, Learning Rate,
   Sequence Length, Neuron, Dropout — training di thread terpisah (UI tidak freeze),
   progress bar + status real-time, tombol Stop
5. Forecasting (prediksi vs aktual pada test set + forecast n-hari ke depan)
6. Evaluasi (MAE, MSE, RMSE, MAPE, R²)
7. Visualisasi Hasil (Loss Training, Loss Validation, Prediksi vs Aktual, Error Distribution)
8. Ringkasan hasil + Export CSV/Excel/PDF

## Yang Masih Perlu Anda Lengkapi untuk Laporan UAS

1. **Ganti dataset sintetis dengan dataset asli** (lihat link di atas) sebelum
   menjalankan eksperimen final yang akan dilaporkan.
2. **Laporan tertulis** (Word/PDF terpisah dari aplikasi ini) berisi: identitas
   kelompok dengan foto wajah anggota — ini tidak bisa dibuat oleh asisten AI
   karena memerlukan foto asli anggota kelompok Anda. Saya bisa bantu membuatkan
   *template* laporan (struktur bab, tabel kosong, dst.) dalam Word jika diminta.
3. Jalankan beberapa skenario eksperimen (variasi transformasi/encoding/PCA untuk
   Project 1, variasi parameter LSTM untuk Project 2) lalu screenshot tabel &
   grafik hasil untuk dimasukkan ke laporan.
