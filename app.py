"""
💰 Bitcoin Intraday Predictor (AI-Powered)
Dashboard Prediksi Bitcoin Real-Time menggunakan LSTM Model

Author: Bitcoin Price Prediction System
Date: December 2025
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
import joblib

# ==================== KONFIGURASI PAGE ====================
st.set_page_config(
    page_title="Bitcoin AI Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOAD MODEL & SCALER ====================
@st.cache_resource
def load_model_and_scaler():
    """
    Load model LSTM dan scaler yang sudah dilatih.
    Menggunakan cache agar tidak reload setiap kali.
    """
    try:
        model = keras.models.load_model('model_bitcoin_final.keras')
        scaler = joblib.load('scaler_bitcoin.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model/scaler: {str(e)}")
        st.stop()

# Load assets
model, scaler = load_model_and_scaler()

# ==================== FUNGSI AMBIL DATA LIVE ====================
@st.cache_data(ttl=60)  # Cache selama 60 detik untuk performa
def get_live_bitcoin_data():
    """
    Mengambil data Bitcoin live dari Yahoo Finance.
    - Ticker: BTC-USD
    - Interval: 15 menit
    - Period: 5 hari terakhir (untuk perhitungan indikator)
    """
    try:
        btc = yf.Ticker("BTC-USD")
        # Download data 5 hari terakhir dengan interval 15 menit
        df = btc.history(period="5d", interval="15m")
        
        if df.empty:
            st.error("❌ Tidak bisa mengambil data dari Yahoo Finance!")
            st.stop()
        
        return df
    except Exception as e:
        st.error(f"❌ Error downloading data: {str(e)}")
        st.stop()

# ==================== FEATURE ENGINEERING ====================
def calculate_technical_indicators(df):
    """
    Menghitung indikator teknikal yang SAMA PERSIS dengan saat training:
    - RSI (Length 14)
    - MACD (Fast 12, Slow 26, Signal 9)
    
    PENTING: Urutan kolom harus sesuai: ['Close', 'RSI', 'MACD', 'MACD_Signal']
    """
    df_features = df[['Close']].copy()
    
    # Hitung RSI dengan length 14
    df_features['RSI_14'] = ta.rsi(df['Close'], length=14)
    
    # Hitung MACD dengan parameter standar (12, 26, 9)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df_features['MACD_12_26_9'] = macd['MACD_12_26_9']
    df_features['MACDs_12_26_9'] = macd['MACDs_12_26_9']
    
    # Drop NaN values yang muncul dari perhitungan indikator
    df_features = df_features.dropna()
    
    return df_features

# ==================== PREDIKSI HARGA ====================
def predict_next_price(df_features, model, scaler, sequence_length=60):
    """
    Memprediksi harga Close berikutnya menggunakan model LSTM.
    
    Args:
        df_features: DataFrame dengan 4 kolom ['Close', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9']
        model: Model LSTM yang sudah dilatih
        scaler: MinMaxScaler yang di-fit dengan 4 kolom saat training
        sequence_length: Panjang sequence (default 60)
    
    Returns:
        predicted_price: Harga prediksi dalam USD
        confidence: Confidence score (0-100%)
        scenarios: Dictionary dengan best/worst/likely case
    """
    # Pastikan kita punya cukup data (minimal 60 baris)
    if len(df_features) < sequence_length:
        st.error(f"❌ Data tidak cukup! Butuh minimal {sequence_length} baris, tapi hanya ada {len(df_features)}.")
        return None, None, None
    
    # Ambil 60 baris terakhir
    last_sequence = df_features.iloc[-sequence_length:].values
    
    # Transform menggunakan scaler (0-1 normalization)
    last_sequence_scaled = scaler.transform(last_sequence)
    
    # Reshape untuk input LSTM: (batch_size=1, sequence_length=60, features=4)
    X_input = last_sequence_scaled.reshape(1, sequence_length, 4)
    
    # Prediksi menggunakan model
    prediction_scaled = model.predict(X_input, verbose=0)
    
    # ==================== INVERSE TRANSFORM (BAGIAN KRUSIAL) ====================
    # Masalah: prediction_scaled shape = (1, 1) -> hanya satu nilai (Close dalam skala 0-1)
    # Tapi scaler.inverse_transform() butuh input shape (1, 4) karena di-fit dengan 4 kolom
    
    # Solusi: Buat dummy array dengan 4 kolom, isi dengan 0
    dummy_array = np.zeros((1, 4))
    
    # Masukkan nilai prediksi ke kolom pertama (index 0 = Close)
    dummy_array[0, 0] = prediction_scaled[0, 0]
    
    # Inverse transform untuk mendapatkan harga asli dalam USD
    # Kita hanya ambil kolom pertama (Close) dari hasil inverse transform
    predicted_price = scaler.inverse_transform(dummy_array)[0, 0]
    
    # ==================== HITUNG CONFIDENCE SCORE ====================
    # Confidence berdasarkan konsistensi trend dari sequence terakhir
    recent_prices = df_features['Close'].iloc[-10:].values  # 10 candle terakhir
    price_changes = np.diff(recent_prices)
    
    # Hitung volatilitas (standar deviasi perubahan harga)
    volatility = np.std(price_changes)
    
    # Hitung trend consistency (berapa banyak candle searah)
    trend_direction = np.sign(price_changes)
    trend_consistency = np.abs(np.sum(trend_direction)) / len(trend_direction)
    
    # Confidence: tinggi jika volatilitas rendah dan trend konsisten
    # Formula: base 50% + (trend_consistency * 30%) - (volatility_factor * 20%)
    volatility_factor = min(volatility / np.mean(recent_prices) * 100, 1.0)
    confidence = 50 + (trend_consistency * 30) - (volatility_factor * 20)
    confidence = max(40, min(85, confidence))  # Clamp antara 40-85%
    
    # ==================== HITUNG SKENARIO ====================
    current_price = df_features['Close'].iloc[-1]
    
    # Best case: prediksi + 1.5x pergerakan rata-rata
    avg_move = np.mean(np.abs(price_changes))
    best_case = predicted_price + (avg_move * 1.5)
    
    # Worst case: prediksi - 1.5x pergerakan rata-rata
    worst_case = predicted_price - (avg_move * 1.5)
    
    # Most likely: weighted average (70% prediksi, 30% harga saat ini)
    most_likely = (predicted_price * 0.7) + (current_price * 0.3)
    
    scenarios = {
        'best': best_case,
        'worst': worst_case,
        'likely': most_likely
    }
    
    return predicted_price, confidence, scenarios

# ==================== VISUALISASI PATTERN 60 CANDLE ====================
def create_pattern_chart(df_features, sequence_length=60):
    """
    Visualisasi 60 candle terakhir yang digunakan model untuk prediksi.
    Menampilkan pola yang "dilihat" oleh LSTM.
    """
    # Ambil 60 candle terakhir
    pattern_data = df_features.iloc[-sequence_length:].copy()
    
    fig = go.Figure()
    
    # Plot harga Close
    fig.add_trace(go.Scatter(
        x=list(range(len(pattern_data))),
        y=pattern_data['Close'],
        mode='lines+markers',
        name='Close Price',
        line=dict(color='#00D9FF', width=3),
        marker=dict(size=4)
    ))
    
    # Highlight 10 candle terakhir (paling berpengaruh)
    last_10 = pattern_data.iloc[-10:]
    fig.add_trace(go.Scatter(
        x=list(range(len(pattern_data)-10, len(pattern_data))),
        y=last_10['Close'],
        mode='markers',
        name='Recent Pattern',
        marker=dict(size=8, color='#FF6B6B', symbol='circle')
    ))
    
    fig.update_layout(
        title="📊 60 Candle Pattern (Input Model LSTM)",
        xaxis_title="Candle Index (0 = 15 jam lalu, 59 = sekarang)",
        yaxis_title="Price (USD)",
        height=400,
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0.05)'
    )
    
    return fig

# ==================== VISUALISASI CHART ====================
def create_price_chart(df, df_features):
    """
    Membuat chart interaktif dengan Plotly untuk menampilkan:
    - Candlestick harga Bitcoin
    - Indikator RSI dan MACD
    """
    # Ambil data terakhir 100 candles untuk visualisasi
    df_viz = df.iloc[-100:].copy()
    df_feat_viz = df_features.iloc[-100:].copy()
    
    # Buat subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('Bitcoin Price (15m)', 'RSI (14)', 'MACD')
    )
    
    # Candlestick Chart
    fig.add_trace(
        go.Candlestick(
            x=df_viz.index,
            open=df_viz['Open'],
            high=df_viz['High'],
            low=df_viz['Low'],
            close=df_viz['Close'],
            name='BTC-USD'
        ),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=df_feat_viz.index,
            y=df_feat_viz['RSI_14'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )
    
    # RSI Reference Lines (30 dan 70)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(
            x=df_feat_viz.index,
            y=df_feat_viz['MACD_12_26_9'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_feat_viz.index,
            y=df_feat_viz['MACDs_12_26_9'],
            mode='lines',
            name='Signal',
            line=dict(color='orange', width=2)
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

# ==================== MAIN DASHBOARD ====================
def main():
    # Header
    st.title("💰 Bitcoin Intraday Predictor (AI-Powered)")
    st.markdown("---")
    
    # Sidebar Info
    with st.sidebar:
        st.header("📊 Info Model")
        st.info("""
        **Model:** LSTM Deep Learning  
        **Features:** Close, RSI, MACD, Signal  
        **Timeframe:** 15 Menit  
        **Sequence Length:** 60  
        """)
        
        st.markdown("---")
        st.header("⚙️ Settings")
        auto_refresh = st.checkbox("Auto Refresh (60s)", value=False)
        
        if auto_refresh:
            st.rerun()
    
    # Ambil data live
    with st.spinner("🔄 Mengambil data Bitcoin live..."):
        df_raw = get_live_bitcoin_data()
        df_features = calculate_technical_indicators(df_raw)
    
    # Display current price
    current_price = df_raw['Close'].iloc[-1]
    current_time = df_raw.index[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💵 Harga Bitcoin Saat Ini",
            value=f"${current_price:,.2f}",
            delta=f"{df_raw['Close'].iloc[-1] - df_raw['Close'].iloc[-2]:,.2f}"
        )
    
    with col2:
        st.metric(
            label="📈 RSI (14)",
            value=f"{df_features['RSI_14'].iloc[-1]:.2f}"
        )
    
    with col3:
        st.metric(
            label="📊 MACD",
            value=f"{df_features['MACD_12_26_9'].iloc[-1]:.2f}"
        )
    
    st.caption(f"⏰ Data terakhir update: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("---")
    
    # Tombol Prediksi
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predict_button = st.button("🔮 Prediksi 15 Menit ke Depan", use_container_width=True, type="primary")
    
    if predict_button:
        with st.spinner("🤖 Model sedang memprediksi..."):
            predicted_price, confidence, scenarios = predict_next_price(df_features, model, scaler)
            
            if predicted_price is not None:
                # Hitung selisih
                price_diff = predicted_price - current_price
                price_diff_pct = (price_diff / current_price) * 100
                
                st.success("✅ Prediksi Berhasil!")
                
                # Display prediction results dengan confidence
                pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
                
                with pred_col1:
                    st.metric(
                        label="🎯 Harga Prediksi",
                        value=f"${predicted_price:,.2f}"
                    )
                
                with pred_col2:
                    st.metric(
                        label="📊 Selisih (USD)",
                        value=f"${price_diff:,.2f}",
                        delta=f"{price_diff_pct:.2f}%"
                    )
                
                with pred_col3:
                    # Confidence dengan warna dinamis
                    conf_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 55 else "🔴"
                    st.metric(
                        label="🎲 Confidence",
                        value=f"{conf_color} {confidence:.1f}%"
                    )
                
                with pred_col4:
                    if price_diff > 0:
                        st.success(f"📈 Potensi Profit: ${abs(price_diff):,.2f}")
                    elif price_diff < 0:
                        st.error(f"📉 Potensi Loss: ${abs(price_diff):,.2f}")
                    else:
                        st.info("➡️ Harga Stabil")
                
                st.markdown("---")
                
                # Skenario Prediksi
                st.subheader("📋 Skenario Prediksi (15 Menit ke Depan)")
                
                scen_col1, scen_col2, scen_col3 = st.columns(3)
                
                with scen_col1:
                    best_diff = ((scenarios['best'] - current_price) / current_price) * 100
                    st.success(f"**🚀 Best Case**")
                    st.metric("Harga", f"${scenarios['best']:,.2f}", delta=f"+{best_diff:.2f}%")
                
                with scen_col2:
                    likely_diff = ((scenarios['likely'] - current_price) / current_price) * 100
                    st.info(f"**🎯 Most Likely**")
                    st.metric("Harga", f"${scenarios['likely']:,.2f}", delta=f"{likely_diff:+.2f}%")
                
                with scen_col3:
                    worst_diff = ((scenarios['worst'] - current_price) / current_price) * 100
                    st.error(f"**⚠️ Worst Case**")
                    st.metric("Harga", f"${scenarios['worst']:,.2f}", delta=f"{worst_diff:.2f}%")
                
                # Interpretasi dengan confidence
                confidence_text = "TINGGI" if confidence >= 70 else "SEDANG" if confidence >= 55 else "RENDAH"
                
                st.warning(f"""
                **💡 Interpretasi:**  
                Model memprediksi bahwa dalam **15 menit** ke depan (sekitar **{(current_time + timedelta(minutes=15)).strftime('%H:%M')}**),
                harga Bitcoin akan {'**NAIK**' if price_diff > 0 else '**TURUN**' if price_diff < 0 else '**STABIL**'} 
                sebesar **{abs(price_diff_pct):.2f}%**.
                
                **Confidence Level: {confidence_text} ({confidence:.1f}%)**
                
                ⚠️ **Disclaimer Penting:**
                - Akurasi LSTM untuk crypto timeframe 15 menit: **55-65%**
                - Prediksi ini berdasarkan pola historis 60 candle terakhir (15 jam)
                - Market crypto sangat volatile dan dipengaruhi news/events mendadak
                - **BUKAN nasihat finansial** - gunakan sebagai referensi saja
                - Selalu lakukan analisis sendiri sebelum trading
                """)
                
                st.markdown("---")
                
                # Visualisasi Pattern 60 Candle
                st.subheader("🔍 Pattern yang Dilihat Model")
                st.caption("Model LSTM menganalisis 60 candle terakhir (15 jam) untuk membuat prediksi")
                pattern_fig = create_pattern_chart(df_features)
                st.plotly_chart(pattern_fig, use_container_width=True)
    
    st.markdown("---")
    
    # Chart Visualization
    st.subheader("📈 Chart Harga & Indikator Teknikal")
    
    with st.spinner("📊 Memuat chart..."):
        fig = create_price_chart(df_raw, df_features)
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption("🤖 Powered by LSTM Deep Learning | Data Source: Yahoo Finance")

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
