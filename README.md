# 💰 Bitcoin Intraday Predictor (AI-Powered)

Dashboard prediksi harga Bitcoin real-time menggunakan **LSTM Deep Learning** dengan timeframe 15 menit.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Fitur Utama

- ✅ **Prediksi Real-Time**: Prediksi harga Bitcoin 15 menit ke depan
- ✅ **Confidence Score**: Tingkat kepercayaan prediksi (40-85%)
- ✅ **Multi-Scenario**: Best case, Most likely, dan Worst case scenarios
- ✅ **Technical Indicators**: RSI (14) dan MACD (12,26,9)
- ✅ **Interactive Charts**: Candlestick, RSI, MACD dengan Plotly
- ✅ **Pattern Visualization**: Visualisasi 60 candle yang dianalisis model
- ✅ **Live Data**: Data real-time dari Yahoo Finance

## 🧠 Model Architecture

**Model**: LSTM (Long Short-Term Memory)
- **Input Features**: 4 kolom (Close, RSI, MACD, MACD Signal)
- **Sequence Length**: 60 candle (15 jam historical data)
- **Timeframe**: 15 menit
- **Normalization**: MinMaxScaler (0-1)

## 📊 Akurasi

- **Timeframe 15 menit**: 55-65% accuracy
- **Confidence scoring** berdasarkan volatilitas dan trend consistency

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.12+
pip
```

### Installation

1. **Clone repository**
```bash
git clone https://github.com/YOUR_USERNAME/bitcoin-predictor.git
cd bitcoin-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run aplikasi**
```bash
streamlit run app.py
```

4. **Buka browser**
```
http://localhost:8501
```

## 📁 File Structure

```
bitcoin-predictor/
├── app.py                      # Main Streamlit application
├── model_bitcoin_final.keras   # Trained LSTM model
├── scaler_bitcoin.pkl          # MinMaxScaler (fitted with 4 features)
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── .gitignore
└── README.md
```

## 🔧 Technical Details

### Feature Engineering

```python
# Indikator yang digunakan
- Close Price (harga penutupan)
- RSI (Relative Strength Index) - Length 14
- MACD (Moving Average Convergence Divergence) - 12,26,9
- MACD Signal Line
```

### Confidence Score Calculation

Confidence score dihitung berdasarkan:
- **Volatilitas**: Standar deviasi perubahan harga
- **Trend Consistency**: Konsistensi arah pergerakan harga
- **Formula**: `50 + (trend_consistency * 30) - (volatility_factor * 20)`

### Prediction Scenarios

- **Best Case**: Prediksi + 1.5x rata-rata pergerakan
- **Most Likely**: 70% prediksi + 30% harga saat ini
- **Worst Case**: Prediksi - 1.5x rata-rata pergerakan

## 📈 Usage Example

1. Dashboard otomatis load harga Bitcoin terkini
2. Lihat indikator RSI dan MACD
3. Klik tombol **"🔮 Prediksi 15 Menit ke Depan"**
4. Lihat hasil:
   - Harga prediksi
   - Confidence score
   - 3 skenario (Best/Likely/Worst)
   - Visualisasi pattern 60 candle

## ⚠️ Disclaimer

- Prediksi ini dibuat oleh model AI untuk **tujuan edukasi dan riset**
- Akurasi untuk crypto timeframe 15 menit: **55-65%**
- Market crypto sangat volatile dan dipengaruhi news/events mendadak
- **BUKAN nasihat finansial** - gunakan sebagai referensi saja
- Selalu lakukan analisis sendiri sebelum trading

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: TensorFlow/Keras (LSTM)
- **Data Source**: Yahoo Finance (yfinance)
- **Technical Analysis**: pandas-ta
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Joblib, Scikit-learn

## 📝 License

MIT License - lihat file LICENSE untuk detail

## 👨‍💻 Author

Developed for thesis project - Bitcoin Price Prediction using LSTM

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📞 Support

Jika ada pertanyaan atau masalah, silakan buat issue di GitHub repository.

---

**⭐ Jangan lupa star repository ini jika bermanfaat!**
