# Bitcoin Price Prediction Dashboard (LSTM)

Repositori ini berisi kode untuk dashboard prediksi harga Bitcoin menggunakan algoritma Deep Learning LSTM (Long Short-Term Memory). Proyek ini dikembangkan sebagai bagian dari tugas skripsi untuk memprediksi harga Bitcoin dalam timeframe 15 menit.

## Website Link
Aplikasi ini sudah dideploy dan dapat diakses melalui:
👉 **[btc-predict-lstm-057.streamlit.app](https://btc-predict-lstm-057.streamlit.app/)**

## Source Code Pelatihan (Jupyter Notebook)
Seluruh tahapan pra-pemrosesan data, desain arsitektur Stacked LSTM, hingga proses pelatihan model (training) dan evaluasinya dapat diakses langsung secara transparan pada Google Colab berikut:
👉 **[Buka Kode Pelatihan Model di Google Colab](https://colab.research.google.com/drive/1-M-irrfB0srhtwqKLY9F_n3gtY1FKn80?usp=sharing)**

## Dataset Information
Model ini dilatih menggunakan data historis BTC/USD. Berikut adalah sumber datanya:
1. **Sumber Asli**: [Bitcoin Historical Data (Kaggle)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
2. **Dataset yang Digunakan**: [Bitcoin Data (Google Drive)](https://drive.google.com/file/d/1uFarNhB68SWrR20oaMs6w-GB5aGUTuSB/view?usp=sharing)
   - Data ini sudah di-resample ke timeframe 15 menit.
   - Rentang data sampai sekitar 4 atau 5 Desember 2025.

## Fitur Utama
- **Prediksi Real-time**: Mengambil data terbaru dari Binance API/Yahoo Finance dan memberikan prediksi 15 menit ke depan.
- **Analisis Teknikal**: Menampilkan indikator RSI (14) dan MACD (12, 26, 9) secara otomatis.
- **Visualisasi Pola**: Menampilkan 60 candle terakhir yang digunakan model sebagai input untuk melakukan prediksi.
- **Skor Kepercayaan**: Memberikan estimasi tingkat kepercayaan model terhadap hasil prediksi.

## Teknologi yang Digunakan
- **Bahasa Pemrograman**: Python 3.12
- **Framework Web**: Streamlit
- **Deep Learning**: TensorFlow / Keras (LSTM)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualisasi**: Plotly

## Cara Menjalankan Lokal
1. Clone repositori ini:
   ```bash
   git clone https://github.com/ENDUGI1/bitcoin-lstm-predictor.git
   ```
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

## Struktur File
- `app.py`: Kode utama aplikasi terminal (Streamlit UI & Logic).
- `config.py`: File konfigurasi parameter model dan aplikasi.
- `model_bitcoin_v1_4features.keras`: File model LSTM yang sudah dilatih.
- `scaler_bitcoin_v1.pkl`: File scaler untuk normalisasi data.
- `requirements.txt`: Daftar library python yang dibutuhkan.

---
**Catatan Penting**: Prediksi harga crypto memiliki risiko tinggi. Dashboard ini dibuat untuk tujuan edukasi dan riset skripsi, bukan sebagai nasihat keuangan profesional.
