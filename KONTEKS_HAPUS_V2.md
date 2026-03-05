# KONTEKS: Hapus Semua Fitur V2 dari Dashboard

## Tujuan
Hapus semua referensi Model V2 (6 features) dari `app.py` dan `config.py` di folder `D:\KULLIAHHHHHH\semester 7\Skripsi\web lstm ta btc\`. Dashboard hanya menggunakan **V1 (4 features: Close, RSI, MACD, MACD Signal)**.

## File yang perlu diubah
1. `app.py` (2534 baris)
2. `config.py` (126 baris)

---

## HAPUS di config.py

```python
# Hapus baris 17-19:
MODEL_V2_PATH = "model_bitcoin_v2_6features.keras"
SCALER_V2_PATH = "scaler_bitcoin_v2.pkl"
MODEL_V2_FEATURES = 6

# Hapus baris 30-31:
ATR_LENGTH = 14

# Hapus baris 92:
CONFIDENCE_ATR_WEIGHT = 15
```

---

## HAPUS di app.py — Fungsi yang harus dihapus sepenuhnya

| Fungsi | Baris | Alasan |
|--------|-------|--------|
| `calculate_atr()` | 77-110 | ATR hanya untuk V2 |
| `download_model_v2_files()` | 734-789 | Download V2 model dari GitHub |
| `load_model_v2()` | 804-820 | Load model V2 |
| `predict_next_price_v2()` | 1243-1312 | Prediksi V2 (6 features) |
| `run_comparison_prediction()` | 1316-1356 | Perbandingan V1 vs V2 |

### Hapus juga pemanggilan:
- Baris 824: `model_v2, scaler_v2 = load_model_v2()` → **HAPUS**

---

## UBAH di app.py — Fungsi yang perlu disederhanakan

### 1. `calculate_technical_indicators()` (baris 998-1047)
- Hapus perhitungan ATR (baris 1029-1032)
- Hapus perhitungan Log Volume (baris 1034-1036)
- Hapus pembuatan `df_model_v2` (baris 1043-1044)
- Return hanya `(df_features, df_model_v1)` bukan 3 nilai
- Update semua tempat yang memanggil fungsi ini (baris 644, 1930) → terima 2 nilai saja

### 2. `create_main_chart()` (baris 1408-1502)
- Ubah dari 4 subplot → 3 subplot (hapus panel ATR)
- Hapus `row_heights=[0.4, 0.2, 0.2, 0.2]` → `[0.4, 0.3, 0.3]`
- Hapus subplot title "ATR (14) - V2 Feature"
- Hapus blok ATR chart (baris 1470-1489)
- Ubah height dari 800 → 700

### 3. `run_backtest()` (baris 630-731)
- Hapus parameter `model_v2, scaler_v2`
- Hapus V2 prediction logic di dalam loop (baris 664-665, 688-701)
- Hapus `'v2'` dari results dict
- Hanya return metrik V1

### 4. Tracker Storage Functions
- `load_tracker_data()` (baris 290-315): Hapus `v2_predictions`, `v2_correct`
- `save_tracker_data()` (baris 317-353): Hapus V2 predictions processing
- `update_actual_prices()` (baris 442-477): Hapus V2 update loop (baris 467-475)
- `create_accuracy_trend_chart()` (baris 503-626): Hapus V2 data processing dan traces

---

## UBAH di app.py — main() function (baris 1505-2533)

### Sidebar (baris 1507-1857)
- **Hapus** Model Selection radio button (baris 1528-1535)
- **Hapus** V2 Model Info block (baris 1548-1561)
- **Hapus** Compare Mode checkbox (baris 1565-1570)
- **Sederhanakan** Performance tracker → hanya V1 (hapus V2 kolom, baris 1607-1629)
- **Hapus** V2 Prediction History (baris 1700-1729)
- **Sederhanakan** Reset tracker data → hanya V1 fields

### Header (baris 1870)
- Ubah `"Advanced LSTM Neural Network • RSI & MACD Strategy • V2 Enhanced"` 
- → `"Advanced LSTM Neural Network • RSI & MACD Strategy"`

### Data Loading (baris 1930)
- Ubah `df_full, df_model_v1, df_model_v2 = calculate_technical_indicators(df_raw)`
- → `df_full, df_model = calculate_technical_indicators(df_raw)`

### Model Selection Logic (baris 1959-1979)
- **Hapus seluruh blok** if/else model_version
- Langsung pakai:
```python
df_model = df_model_v1  # (sekarang cuma df_model dari fungsi)
model_active = model
scaler_active = scaler
predict_func = predict_next_price
```

### Prediction Core (baris 2091-2208)
- **Hapus** comparison mode block (baris 2107-2117)
- Langsung jalankan V1 prediction saja
- **Hapus** V2 tracking di baris 2151-2154 (selalu track sebagai V1)

### Confidence Breakdown (baris 2270-2274)
- **Hapus** V2 ATR penalty info
- Langsung tampilkan V1 breakdown saja

### Backtesting call (baris 1759-1762)
- Hapus `model_v2, scaler_v2` dari parameter

### Comparison Results Display (baris 2356-2406)
- **Hapus seluruh blok**

### Accuracy Trend (baris 2408-2432)
- Ubah caption dari "V1 vs V2" → cukup "Model Performance"
- Hapus V2 references

### Backtest Results Display (baris 2434-2500)
- Hapus V2 card dan comparison card
- Hanya tampilkan V1 results

---

## Yang TETAP (jangan dihapus!)
- ✅ `predict_next_price()` — fungsi prediksi V1
- ✅ `create_confidence_gauge()` — gauge chart
- ✅ `create_pattern_chart()` — pattern 60 candle
- ✅ Telegram alert system
- ✅ TradingView widget
- ✅ CSV Export
- ✅ Seluruh CSS cyberpunk
- ✅ Live clock
- ✅ Data source badge (Binance/Yahoo)
- ✅ Performance tracker (V1 only)
- ✅ Backtesting (V1 only)
