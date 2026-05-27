import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import joblib
import base64
import streamlit.components.v1 as components
import plotly.graph_objects as go
import logging
import json
import os

# Import configuration
import config

# Design system (Quiet Quant — see styles.py)
import styles
from styles import TOKENS, PLOTLY_THEME, PLOTLY_AXIS, ICONS

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)
logger.info(" Bitcoin LSTM Dashboard started")

# ==================== KONFIGURASI PAGE ====================
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="bitcoin-btc-logo.png", # Custom Bitcoin logo
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ... (CSS Injection remains checks out, skip to Feature Engineering)

# ==================== MANUAL INDICATOR FUNCTIONS (NO DEPENDENCIES) ====================
def calculate_rsi(series, period=14):
    """
    Calculate RSI using Wilder's Smoothing Method.
    This matches pandas_ta library used in model training.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # First average using SMA for initial value
    first_avg_gain = gain.iloc[:period].mean()
    first_avg_loss = loss.iloc[:period].mean()
    
    # Wilder's Smoothing: avg = (prev_avg * (n-1) + current) / n
    avg_gain = gain.copy()
    avg_loss = loss.copy()
    
    avg_gain.iloc[:period] = np.nan
    avg_loss.iloc[:period] = np.nan
    avg_gain.iloc[period-1] = first_avg_gain
    avg_loss.iloc[period-1] = first_avg_loss
    
    for i in range(period, len(series)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD manually using Pandas"""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# ==================== CUSTOM CSS — QUIET QUANT DESIGN SYSTEM ====================
# All styling lives in styles.py (tokens, CSS, Plotly theme, SVG icons, helpers).
# This wrapper keeps the original API stable; the implementation is delegated.
def inject_custom_css():
    styles.inject()

inject_custom_css()

# ==================== PERSISTENT TRACKER STORAGE ====================
TRACKER_FILE = "tracker_data.json"

def load_tracker_data():
    """Load tracker data from JSON file"""
    try:
        if os.path.exists(TRACKER_FILE):
            with open(TRACKER_FILE, 'r') as f:
                data = json.load(f)
                logger.info(f"Tracker data loaded: {len(data.get('predictions', []))} predictions")
                return data
        else:
            logger.info("No existing tracker data found, starting fresh")
            return {
                'predictions': [],
                'correct': 0,
                'last_actual_price': None
            }
    except Exception as e:
        logger.error(f"Error loading tracker data: {str(e)}, starting fresh")
        return {
            'predictions': [],
            'correct': 0,
            'last_actual_price': None
        }

def save_tracker_data(tracker_data):
    """Save tracker data to JSON file"""
    try:
        # Convert datetime objects to strings for JSON serialization
        data_to_save = {
            'predictions': [],
            'correct': tracker_data.get('correct', 0),
            'last_actual_price': tracker_data.get('last_actual_price')
        }
        
        # Convert predictions with datetime to serializable format
        for pred in tracker_data.get('predictions', []):
            pred_copy = pred.copy()
            if isinstance(pred_copy.get('timestamp'), datetime):
                pred_copy['timestamp'] = pred_copy['timestamp'].isoformat()
            data_to_save['predictions'].append(pred_copy)
        
        # Keep only last 100 predictions to avoid file bloat
        data_to_save['predictions'] = data_to_save['predictions'][-100:]
        
        with open(TRACKER_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        logger.info(f"Tracker data saved: {len(data_to_save['predictions'])} predictions")
        return True
    except Exception as e:
        logger.error(f"Error saving tracker data: {str(e)}")
        return False

# ==================== ACCURACY METRICS CALCULATION ====================
def calculate_accuracy_metrics(predictions_list):
    """
    Calculate MAE, RMSE, and directional accuracy from prediction history.
    Returns dict with metrics or None if insufficient data.
    """
    if not predictions_list or len(predictions_list) < 2:
        return None
    
    # Filter predictions that have actual prices
    valid_predictions = [p for p in predictions_list if p.get('actual_price') is not None]
    
    if len(valid_predictions) < 2:
        return None
    
    predicted_prices = [p['predicted_price'] for p in valid_predictions]
    actual_prices = [p['actual_price'] for p in valid_predictions]
    
    # Calculate MAE (Mean Absolute Error)
    errors = [abs(pred - actual) for pred, actual in zip(predicted_prices, actual_prices)]
    mae = np.mean(errors)
    
    # Calculate RMSE (Root Mean Square Error)
    squared_errors = [(pred - actual) ** 2 for pred, actual in zip(predicted_prices, actual_prices)]
    rmse = np.sqrt(np.mean(squared_errors))
    
    # Calculate Directional Accuracy
    correct_directions = 0
    for p in valid_predictions:
        predicted_direction = p.get('direction', 'unknown')
        current_price = p.get('current_price', 0)
        actual_price = p.get('actual_price', 0)
        actual_direction = 'up' if actual_price > current_price else 'down'
        
        if predicted_direction == actual_direction:
            correct_directions += 1
    
    directional_accuracy = (correct_directions / len(valid_predictions)) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'directional_accuracy': directional_accuracy,
        'total_predictions': len(valid_predictions)
    }


def create_confidence_gauge(confidence, title="Confidence"):
    """
    Confidence gauge (Quiet Quant theme).
    - Bar uses the single accent (clay) so it reads as a *signal*, not a separate color.
    - Steps are desaturated direction tints (down / warning / up) — no neon red/green.
    - Threshold marker removed: redundant with the bar value itself.
    """
    # Pick the active segment color so the bar always reads as the dominant tone
    if confidence >= 70:
        bar_color = TOKENS["up"]
    elif confidence >= 55:
        bar_color = TOKENS["warning"]
    else:
        bar_color = TOKENS["down"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': title,
            'font': {'size': 12, 'color': TOKENS["text_muted"],
                     'family': "Manrope, sans-serif"},
        },
        number={
            'suffix': "%",
            'font': {'size': 36, 'color': TOKENS["text_primary"],
                     'family': "JetBrains Mono, monospace"},
            'valueformat': '.0f',
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': TOKENS["text_faint"],
                'tickfont': {'family': 'JetBrains Mono, monospace', 'size': 10,
                             'color': TOKENS["text_muted"]},
                'tickvals': [0, 25, 50, 75, 100],
            },
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'bordercolor': 'rgba(0,0,0,0)',
            'steps': [
                {'range': [0, 55],   'color': TOKENS["down_soft"]},
                {'range': [55, 70],  'color': TOKENS["warning_soft"]},
                {'range': [70, 100], 'color': TOKENS["up_soft"]},
            ],
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': TOKENS["text_primary"], 'family': "Manrope, sans-serif"},
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ==================== ACTUAL PRICE UPDATE SYSTEM ====================
def update_actual_prices(tracker_data, current_df):
    """
    Auto-update actual prices for predictions older than 15 minutes.
    Returns number of predictions updated.
    """
    updated_count = 0
    current_time = datetime.now()
    current_price = current_df['Close'].iloc[-1]
    
    for pred in tracker_data.get('predictions', []):
        if pred.get('actual_price') is None:
            pred_time = pred.get('timestamp')
            if isinstance(pred_time, str):
                pred_time = datetime.fromisoformat(pred_time)
            
            # Check if prediction is older than 15 minutes
            if pred_time and (current_time - pred_time).total_seconds() >= 900: # 15 min = 900 sec
                pred['actual_price'] = current_price
                updated_count += 1
    
    return updated_count


def manual_update_all_actual_prices(tracker_data, current_price):
    """
    Manually update ALL predictions with current price (for testing/demo).
    Returns number of predictions updated.
    """
    updated_count = 0
    
    for pred in tracker_data.get('predictions', []):
        if pred.get('actual_price') is None:
            pred['actual_price'] = current_price
            updated_count += 1
    
    return updated_count


# ==================== ACCURACY TREND CHART ====================
def create_accuracy_trend_chart(tracker_data):
    """
    Create trend chart showing model accuracy over time.
    Shows directional accuracy, MAE, and RMSE as predictions accumulate.
    """
    from plotly.subplots import make_subplots
    
    preds = [p for p in tracker_data.get('predictions', []) if p.get('actual_price') is not None]
    
    if len(preds) < 2:
        return None
    
    # Calculate cumulative metrics
    def calc_cumulative_metrics(predictions):
        counts = []
        dir_acc = []
        mae_vals = []
        rmse_vals = []
        
        for i in range(2, len(predictions) + 1):
            subset = predictions[:i]
            counts.append(i)
            
            # Directional accuracy
            correct = sum(1 for p in subset if p.get('direction') == ('up' if p['actual_price'] > p['current_price'] else 'down'))
            dir_acc.append((correct / i) * 100)
            
            # MAE
            errors = [abs(p['predicted_price'] - p['actual_price']) for p in subset]
            mae_vals.append(np.mean(errors))
            
            # RMSE
            squared_errors = [(p['predicted_price'] - p['actual_price']) ** 2 for p in subset]
            rmse_vals.append(np.sqrt(np.mean(squared_errors)))
        
        return counts, dir_acc, mae_vals, rmse_vals
    
    # Three stacked panels — sequential, low-density. Subtitles use eyebrow-style caps.
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('DIRECTIONAL ACCURACY', 'MEAN ABSOLUTE ERROR', 'ROOT MEAN SQUARE ERROR'),
        vertical_spacing=0.12,
        row_heights=[0.34, 0.33, 0.33]
    )

    if len(preds) >= 2:
        counts, dir_vals, mae_vals, rmse_vals = calc_cumulative_metrics(preds)

        # All three use the same accent — restraint. Marker only at last point.
        marker_emphasis = dict(size=6, color=TOKENS["accent"],
                               line=dict(color=TOKENS["surface_base"], width=1))

        fig.add_trace(go.Scatter(
            x=counts, y=dir_vals,
            name='Directional Accuracy',
            line=dict(color=TOKENS["accent"], width=1.6),
            mode='lines',
            showlegend=False,
            hovertemplate='<b>%{y:.1f}%</b> · n=%{x}<extra></extra>',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[counts[-1]], y=[dir_vals[-1]],
            mode='markers', marker=marker_emphasis, showlegend=False,
            hoverinfo='skip',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=counts, y=mae_vals, name='MAE',
            line=dict(color=TOKENS["text_secondary"], width=1.4),
            mode='lines', showlegend=False,
            hovertemplate='<b>$%{y:,.2f}</b> · n=%{x}<extra></extra>',
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[counts[-1]], y=[mae_vals[-1]],
            mode='markers', marker=marker_emphasis, showlegend=False, hoverinfo='skip',
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=counts, y=rmse_vals, name='RMSE',
            line=dict(color=TOKENS["text_secondary"], width=1.4),
            mode='lines', showlegend=False,
            hovertemplate='<b>$%{y:,.2f}</b> · n=%{x}<extra></extra>',
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=[counts[-1]], y=[rmse_vals[-1]],
            mode='markers', marker=marker_emphasis, showlegend=False, hoverinfo='skip',
        ), row=3, col=1)

    # Apply shared axis style to every subplot
    for r in (1, 2, 3):
        fig.update_xaxes(**PLOTLY_AXIS, row=r, col=1)
        fig.update_yaxes(**PLOTLY_AXIS, row=r, col=1)

    fig.update_xaxes(title_text="Predictions verified", row=3, col=1)
    fig.update_yaxes(title_text="Accuracy %",  row=1, col=1, ticksuffix="%")
    fig.update_yaxes(title_text="USD", row=2, col=1, tickprefix="$")
    fig.update_yaxes(title_text="USD", row=3, col=1, tickprefix="$")

    # Subtitle styling — eyebrow caps in muted mono
    fig.update_annotations(
        font=dict(family="JetBrains Mono, monospace", size=10, color=TOKENS["text_muted"]),
        xanchor='left', x=0,
    )

    fig.update_layout(
        **PLOTLY_THEME,
        height=620,
        showlegend=False,
    )
    return fig


# ==================== BACKTESTING SYSTEM ====================
def fetch_binance_historical(start_date, end_date, symbol="BTCUSDT", interval="15m"):
    """
    Fetch historical OHLCV from Binance data-api with pagination.
    Supports any historical date range — no 60-day yfinance limit.
    Logic mirrors the Jupyter notebook approach exactly.
    """
    import requests as _req

    url = "https://data-api.binance.vision/api/v3/klines"
    start_milli   = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_milli     = int(pd.Timestamp(end_date).timestamp() * 1000)
    current_start = start_milli
    all_klines    = []

    logger.info(f"Fetching Binance historical: {start_date} → {end_date}")

    while current_start < end_milli:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": current_start,
            "endTime":   end_milli,
            "limit":     1000,
        }
        try:
            resp = _req.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Binance API request failed: {e}")
            break

        if not data:
            break

        all_klines.extend(data)
        current_start = data[-1][0] + 1  # advance to avoid duplicate candle

    if not all_klines:
        return None

    cols = [
        'OpenTime','Open','High','Low','Close','Volume',
        'CloseTime','QuoteVolume','Trades','TakerBuyBase','TakerBuyQuote','Ignore'
    ]
    df = pd.DataFrame(all_klines, columns=cols)
    df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
    df.set_index('OpenTime', inplace=True)
    for col in ['Open','High','Low','Close','Volume']:
        df[col] = df[col].astype(float)

    df = df[['Open','High','Low','Close','Volume']]
    logger.info(f"Binance historical fetched: {len(df)} klines")
    return df


def run_backtest(start_date, end_date, model_v1, scaler_v1):
    """
    Run backtesting using Binance API — no date-range restriction.
    Returns comprehensive metrics and prediction history.
    """
    try:
        # --- Ambil data dari Binance (bukan yfinance, bebas batas tanggal) ---
        df_hist = fetch_binance_historical(start_date, end_date)

        if df_hist is None or df_hist.empty or len(df_hist) < 100:
            return None, (
                "Insufficient historical data for selected date range. "
                "Pastikan koneksi internet aktif dan rentang tanggal memiliki cukup data."
            )

        logger.info(f"Backtest dataset: {len(df_hist)} rows "
                    f"({df_hist.index[0]} → {df_hist.index[-1]})")

        # Hitung indikator teknikal (tanpa @st.cache_data karena data lokal)
        df_full, df_model = calculate_technical_indicators(df_hist)

        if len(df_model) < 62:
            return None, "Data setelah kalkulasi indikator tidak cukup (butuh minimal 62 baris)."

        # ── VECTORIZED SLIDING WINDOW + BATCH PREDICTION ─────────────────
        # Rename kolom agar cocok dengan nama kolom saat scaler di-fit
        df_scaled_input = df_model.copy()
        if len(df_scaled_input.columns) == 4:
            df_scaled_input.columns = ['Close', 'RSI', 'MACD', 'MACD_Signal']

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled_data = scaler_v1.transform(df_scaled_input)

        time_step = 60
        n_samples = len(scaled_data) - time_step

        # Build seluruh sliding window sekaligus (tanpa loop) — O(1) overhead
        idx = np.arange(time_step)[None, :] + np.arange(n_samples)[:, None]
        X_all = scaled_data[idx]  # shape: (n_samples, 60, 4)

        # Satu kali model.predict untuk semua sample — 40-100x lebih cepat
        logger.info(f"Batched prediction: {n_samples} samples...")
        pred_scaled = model_v1.predict(X_all, batch_size=512, verbose=0)

        # Inverse-transform: hanya kolom Close (index 0)
        dummy = np.zeros((len(pred_scaled), 4))
        dummy[:, 0] = pred_scaled.flatten()
        y_pred = scaler_v1.inverse_transform(dummy)[:, 0]

        # ── HITUNG METRIK ─────────────────────────────────────────────────
        results = {'predictions': [], 'directional_correct': 0, 'total': 0}

        for i in range(n_samples):
            # FIX: gunakan df_model (sudah dropna) bukan df_hist
            # Ini menghilangkan bug alignment akibat warmup indikator teknikal
            actual_price  = df_model['Close'].iloc[i + time_step]
            current_price = df_model['Close'].iloc[i + time_step - 1]
            pred          = float(y_pred[i])

            error = abs(pred - actual_price)

            # FIX: directional accuracy tanpa threshold — murni np.sign
            direction        = 'up' if pred > current_price else 'down'
            actual_direction = 'up' if actual_price > current_price else 'down'
            is_correct       = direction == actual_direction

            results['predictions'].append({
                'predicted':         pred,
                'actual':            actual_price,
                'error':             error,
                'direction_correct': is_correct,
            })
            if is_correct:
                results['directional_correct'] += 1
            results['total'] += 1

        metrics = {}
        preds   = results['predictions']

        if preds:
            errors = [p['error'] for p in preds]
            metrics['v1'] = {
                'total_predictions':    len(preds),
                'directional_accuracy': (results['directional_correct'] / results['total']) * 100,
                'mae':                  np.mean(errors),
                'rmse':                 np.sqrt(np.mean([e**2 for e in errors])),
                'min_error':            np.min(errors),
                'max_error':            np.max(errors),
                'median_error':         np.median(errors),
            }
        else:
            metrics['v1'] = None

        return metrics, None

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, str(e)

# ==================== LOAD MODEL & SCALER ====================
@st.cache_resource
def load_model_and_scaler():
    """Load pretrained LSTM model and scaler (4 features)"""
    try:
        model = tf.keras.models.load_model(config.MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        logger.info(" Model (4 features) loaded successfully")
        return model, scaler
    except Exception as e:
        st.error(f" Error loading Model: {str(e)}")
        st.stop()

# Load model
model, scaler = load_model_and_scaler()

# ==================== FUNGSI AMBIL DATA LIVE ====================
def get_binance_btc_data():
    """
    Fetch real-time BTCUSDT data from Binance Public API.
    Returns DataFrame with columns: Open, High, Low, Close, Volume
    Includes retry logic for reliability.
    """
    import requests
    import time
    
    max_retries = 3
    retry_delay = 1 # seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching Bitcoin data from Binance (attempt {attempt + 1}/{max_retries})")
            
            url = f"{config.BINANCE_BASE_URL}/api/v3/klines"
            params = {
                "symbol": config.BINANCE_SYMBOL,
                "interval": config.BINANCE_INTERVAL,
                "limit": config.BINANCE_LIMIT
            }
            
            # 6s timeout — long enough for slow connections, short enough to fail
            # fast and fall through to yfinance instead of hanging the whole render.
            response = requests.get(url, params=params, timeout=6)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning("Binance API returned empty data")
                continue
            
            # Convert to DataFrame
            # Binance klines format: [OpenTime, Open, High, Low, Close, Volume, CloseTime, ...]
            df = pd.DataFrame(data, columns=[
                'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
                'CloseTime', 'QuoteVolume', 'Trades', 'TakerBuyBase', 'TakerBuyQuote', 'Ignore'
            ])
            
            # Convert to proper types
            df['Open'] = df['Open'].astype(float)
            df['High'] = df['High'].astype(float)
            df['Low'] = df['Low'].astype(float)
            df['Close'] = df['Close'].astype(float)
            df['Volume'] = df['Volume'].astype(float)
            
            # Convert timestamp to datetime index
            df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
            df.set_index('OpenTime', inplace=True)
            
            # Keep only OHLCV columns (same format as yfinance)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            logger.info(f" Binance data fetched successfully: {len(df)} candles")
            return df
            
        except requests.exceptions.Timeout:
            logger.warning(f"Binance API timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API request failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Error processing Binance data: {str(e)}")
            break
    
    logger.warning("All Binance API attempts failed")
    return None




def get_live_bitcoin_data():
    """
    Fetch live Bitcoin data with fallback mechanism.
    Primary: Binance API (real-time)
    Fallback: Yahoo Finance (delayed ~15 min)
    """
    import time
    
    # Try Binance first if configured
    data_source = getattr(config, 'DATA_SOURCE', 'yfinance')
    
    if data_source == "binance":
        df = get_binance_btc_data()
        if df is not None and not df.empty:
            logger.info(" Using Binance data (real-time)")
            # Track data source for UI indicator
            st.session_state['data_source_used'] = 'binance'
            return df
        else:
            logger.warning(" Binance failed, falling back to yfinance...")
    
    # Track that we're using yfinance
    st.session_state['data_source_used'] = 'yfinance'
    
    # Fallback to yfinance
    logger.info(f"Fetching Bitcoin data from yfinance: {config.TICKER_SYMBOL}, Period: {config.DATA_PERIOD}, Interval: {config.DATA_INTERVAL}")
    
    for i in range(config.MAX_RETRIES):
        try:
            # Try fetching with yf.download wrapper which proved more stable
            df = yf.download(config.TICKER_SYMBOL, period=config.DATA_PERIOD, interval=config.DATA_INTERVAL, progress=False)
            
            if not df.empty:
                # Flatten MultiIndex columns if present (yf.download returns MultiIndex)
                # This prevents "unsupported format string passed to Series.__format__" errors
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                logger.info(f" yfinance data fetched successfully: {len(df)} candles")
                return df
                
            # If empty, wait and retry
            logger.warning(f"Attempt {i+1}/{config.MAX_RETRIES}: Empty data received, retrying...")
            time.sleep(config.RETRY_DELAY_SECONDS)
            
        except Exception as e:
            logger.error(f"Attempt {i+1}/{config.MAX_RETRIES} failed: {str(e)}")
            if i == config.MAX_RETRIES - 1: # Last attempt
                st.error(f" Failed to fetch data after {config.MAX_RETRIES} attempts: {str(e)}")
                st.stop()
            time.sleep(1)
            
    st.error(" Failed to fetch data from Yahoo Finance (Empty Data)")
    st.stop()

# ==================== DATA VALIDATION ====================
def validate_data_for_prediction(df, min_rows=config.MIN_DATA_ROWS):
    """
    Validate if data is suitable for LSTM prediction
    Returns: (is_valid: bool, message: str)
    """
    logger.info("Validating data for prediction...")
    
    # Check 1: DataFrame not empty
    if df is None or df.empty:
        logger.error("Validation failed: Empty DataFrame")
        return False, " Data kosong. Tidak bisa melakukan prediksi."
    
    # Check 2: Required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f" Kolom yang diperlukan tidak ada: {', '.join(missing_cols)}"
    
    # Check 3: Sufficient data points
    if len(df) < min_rows:
        return False, f" Data tidak cukup untuk prediksi. Diperlukan minimal {min_rows} candle, tersedia {len(df)}."
    
    # Check 4: No NaN values in critical columns
    if df[required_cols].isnull().any().any():
        nan_cols = df[required_cols].columns[df[required_cols].isnull().any()].tolist()
        return False, f" Data mengandung nilai kosong (NaN) pada kolom: {', '.join(nan_cols)}"
    
    # Check 5: Price values are positive
    if (df['Close'] <= 0).any():
        return False, " Data harga mengandung nilai negatif atau nol. Data tidak valid."
    
    # Check 6: Reasonable price range (sanity check for BTC)
    min_price = df['Close'].min()
    max_price = df['Close'].max()
    if min_price < config.MIN_PRICE_USD or max_price > config.MAX_PRICE_USD:
        logger.error(f"Validation failed: Price out of range (${min_price:,.0f} - ${max_price:,.0f})")
        return False, f" Harga di luar rentang wajar (${min_price:,.0f} - ${max_price:,.0f}). Kemungkinan data corrupt."
    
    # All checks passed
    logger.info(" Data validation passed")
    return True, " Data valid untuk prediksi."

# ==================== FEATURE ENGINEERING ====================
@st.cache_data(ttl=config.CACHE_TTL_INDICATORS, show_spinner=False,
               hash_funcs={pd.DataFrame: lambda df: df.to_json()})
def calculate_technical_indicators(df):
    """
    Calculate technical indicators for LSTM model:
    4 features: Close, RSI, MACD, Signal
    
    Cached for 5 minutes to improve performance.
    Returns: (df_features, df_model)
    """
    logger.info(f"Calculating technical indicators for {len(df)} candles")
    
    df_features = df.copy()
    
    # RSI (14) using manual function
    df_features['RSI_14'] = calculate_rsi(df_features['Close'], period=config.RSI_LENGTH)
    logger.debug(f"RSI calculated: {df_features['RSI_14'].iloc[-1]:.2f}")
    
    # MACD (12, 26, 9) using manual function
    macd_line, signal_line, histogram = calculate_macd(
        df_features['Close'], 
        fast=config.MACD_FAST, 
        slow=config.MACD_SLOW, 
        signal=config.MACD_SIGNAL
    )
    
    df_features['MACD_12_26_9'] = macd_line
    df_features['MACDs_12_26_9'] = signal_line
    logger.debug(f"MACD calculated: {macd_line.iloc[-1]:.4f}")
    
    df_features = df_features.dropna()
    
    # Model: 4 features (Close, RSI, MACD, Signal)
    df_model = df_features[['Close', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9']].copy()
    
    logger.info(f"Technical indicators calculated. {len(df_model)} rows")
    return df_features, df_model

# ==================== TELEGRAM ALERT SYSTEM ====================
def send_telegram_message(bot_token, chat_id, message):
    """Send message via Telegram Bot API"""
    if not bot_token or not chat_id:
        logger.warning("Telegram credentials not set")
        return False, "Bot Token or Chat ID not configured"
    
    try:
        import requests
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=payload, timeout=5)
        
        if response.status_code == 200:
            logger.info(f"Telegram message sent successfully to {chat_id}")
            return True, "Message sent!"
        else:
            logger.error(f"Telegram API error: {response.text}")
            return False, f"API Error: {response.status_code}"
            
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {str(e)}")
        return False, str(e)

def check_and_send_alerts(bot_token, chat_id, rsi_val, macd_val, signal_val, current_price, alert_settings):
    """Check conditions and send alerts if triggered"""
    if not bot_token or not chat_id:
        return
    
    alerts_sent = []
    
    # RSI Overbought Alert
    if alert_settings.get('rsi_overbought', False) and rsi_val > config.ALERT_RSI_OVERBOUGHT:
        message = f"""
 <b>RSI OVERBOUGHT ALERT</b>

 RSI: {rsi_val:.1f} (>{config.ALERT_RSI_OVERBOUGHT})
 BTC Price: ${current_price:,.2f}

 Market mungkin jenuh beli. Potensi koreksi turun.

 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        success, _ = send_telegram_message(bot_token, chat_id, message)
        if success:
            alerts_sent.append("RSI Overbought")
    
    # RSI Oversold Alert
    if alert_settings.get('rsi_oversold', False) and rsi_val < config.ALERT_RSI_OVERSOLD:
        message = f"""
 <b>RSI OVERSOLD ALERT</b>

 RSI: {rsi_val:.1f} (<{config.ALERT_RSI_OVERSOLD})
 BTC Price: ${current_price:,.2f}

 Market mungkin jenuh jual. Potensi rebound naik.

 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        success, _ = send_telegram_message(bot_token, chat_id, message)
        if success:
            alerts_sent.append("RSI Oversold")
    
    # MACD Crossover Alert
    if alert_settings.get('macd_crossover', False):
        hist = macd_val - signal_val
        if abs(hist) < 5: # Close to crossover
            trend = "BULLISH " if hist > 0 else "BEARISH "
            message = f"""
 <b>MACD SIGNAL</b>

 MACD: {macd_val:.2f}
 Signal: {signal_val:.2f}
 Histogram: {hist:.2f}

{trend}

 BTC Price: ${current_price:,.2f}

 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
            success, _ = send_telegram_message(bot_token, chat_id, message)
            if success:
                alerts_sent.append("MACD Signal")
    
    return alerts_sent

# ==================== ASSETS ====================
def get_bitcoin_logo_base64():
    # SVG string content
    svg = """<svg xmlns="http://www.w3.org/2000/svg" xml:space="preserve" width="100%" height="100%" version="1.1" shape-rendering="geometricPrecision" text-rendering="geometricPrecision" image-rendering="optimizeQuality" fill-rule="evenodd" clip-rule="evenodd" viewBox="0 0 4091.27 4091.73" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xodm="http://www.corel.com/coreldraw/odm/2003"><g id="Layer_x0020_1"><metadata id="CorelCorpID_0Corel-Layer"/><g id="_1421344023328"><path fill="#F7931A" fill-rule="nonzero" d="M4030.06 2540.77c-273.24,1096.01 -1383.32,1763.02 -2479.46,1489.71 -1095.68,-273.24 -1762.69,-1383.39 -1489.33,-2479.31 273.12,-1096.13 1383.2,-1763.19 2479,-1489.95 1096.06,273.24 1763.03,1383.51 1489.76,2479.57l0.02 -0.02z"/><path fill="white" fill-rule="nonzero" d="M2947.77 1754.38c40.72,-272.26 -166.56,-418.61 -450,-516.24l91.95 -368.8 -224.5 -55.94 -89.51 359.09c-59.02,-14.72 -119.63,-28.59 -179.87,-42.34l90.16 -361.46 -224.36 -55.94 -92 368.68c-48.84,-11.12 -96.81,-22.11 -143.35,-33.69l0.26 -1.16 -309.59 -77.31 -59.72 239.78c0,0 166.56,38.18 163.05,40.53 90.91,22.69 107.35,82.87 104.62,130.57l-104.74 420.15c6.26,1.59 14.38,3.89 23.34,7.49 -7.49,-1.86 -15.46,-3.89 -23.73,-5.87l-146.81 588.57c-11.11,27.62 -39.31,69.07 -102.87,53.33 2.25,3.26 -163.17,-40.72 -163.17,-40.72l-111.46 256.98 292.15 72.83c54.35,13.63 107.61,27.89 160.06,41.3l-92.9 373.03 224.24 55.94 92 -369.07c61.26,16.63 120.71,31.97 178.91,46.43l-91.69 367.33 224.51 55.94 92.89 -372.33c382.82,72.45 670.67,43.24 791.83,-303.02 97.63,-278.78 -4.86,-439.58 -206.26,-544.44 146.69,-33.83 257.18,-130.31 286.64,-329.61l-0.07 -0.05zm-512.93 719.26c-69.38,278.78 -538.76,128.08 -690.94,90.29l123.28 -494.2c152.17,37.99 640.17,113.17 567.67,403.91zm69.43 -723.3c-63.29,253.58 -453.96,124.75 -580.69,93.16l111.77 -448.21c126.73,31.59 534.85,90.55 468.94,355.05l-0.02 0z"/></g></g></svg>"""
    return base64.b64encode(svg.encode('utf-8')).decode('utf-8')


# ==================== TRADINGVIEW INTERACTIVE CHART ====================
def render_tradingview_widget():
    """
    Render TradingView Advanced Chart widget for interactive charting.
    Features: Real-time data, drawing tools, technical indicators.
    """
    # Get config values with fallbacks
    symbol = getattr(config, 'TRADINGVIEW_SYMBOL', 'BINANCE:BTCUSDT')
    theme = getattr(config, 'TRADINGVIEW_THEME', 'dark')
    height = getattr(config, 'TRADINGVIEW_HEIGHT', 500)
    allow_symbol_change = str(getattr(config, 'TRADINGVIEW_ALLOW_SYMBOL_CHANGE', False)).lower()
    
    widget_html = f'''
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="height:{height}px; width:100%;">
        <div id="tradingview_btc" style="height:100%; width:100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({{
            "autosize": true,
            "symbol": "{symbol}",
            "interval": "15",
            "timezone": "Etc/UTC",
            "theme": "{theme}",
            "style": "1",
            "locale": "en",
            "enable_publishing": false,
            "allow_symbol_change": {allow_symbol_change},
            "hide_top_toolbar": false,
            "hide_legend": false,
            "save_image": true,
            "container_id": "tradingview_btc",
            "studies": [
                "RSI@tv-basicstudies",
                "MACD@tv-basicstudies"
            ]
        }});
        </script>
    </div>
    <!-- TradingView Widget END -->
    '''
    
    components.html(widget_html, height=height + 20)
    logger.info("TradingView widget rendered successfully")




# ==================== LSTM PREDICTION ====================
def predict_next_price(df_model, model, scaler, sequence_length=config.SEQUENCE_LENGTH):
    logger.debug(f"Starting LSTM prediction with sequence length: {sequence_length}")
    
    if len(df_model) < sequence_length:
        logger.error(f"Insufficient data: {len(df_model)} rows (need {sequence_length})")
        st.error(f" Data insufficient! Need {sequence_length} rows.")
        return None, None, None
    
    # Ambil 60 data terakhir dan ubah nama kolom untuk menghindari warning sklearn
    import warnings
    last_sequence_df = df_model.iloc[-sequence_length:].copy()
    
    # Fallback to safely renaming columns to what scaler expects
    if len(last_sequence_df.columns) == 4:
        last_sequence_df.columns = ['Close', 'RSI', 'MACD', 'MACD_Signal']
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        last_sequence_scaled = scaler.transform(last_sequence_df)
        
    X_input = last_sequence_scaled.reshape(1, sequence_length, 4)
    
    logger.debug(f"Input shape: {X_input.shape}")
    prediction_scaled = model.predict(X_input, verbose=0)
    
    # Inverse Transform Logic
    dummy_array = np.zeros((1, 4))
    dummy_array[0, 0] = prediction_scaled[0, 0]
    predicted_price = scaler.inverse_transform(dummy_array)[0, 0]
    
    logger.debug(f"Raw prediction: ${predicted_price:,.2f}")
    
    # Confidence Calculation (Standard deviation/Trend based)
    recent_prices = df_model['Close'].iloc[-10:].values
    price_changes = np.diff(recent_prices)
    volatility = np.std(price_changes)
    trend_consistency = np.abs(np.sum(np.sign(price_changes))) / len(price_changes)
    
    volatility_factor = min(volatility / np.mean(recent_prices) * 100, 1.0)
    confidence = config.CONFIDENCE_BASE + (trend_consistency * config.CONFIDENCE_TREND_WEIGHT) - (volatility_factor * config.CONFIDENCE_VOLATILITY_WEIGHT)
    confidence = max(config.CONFIDENCE_MIN, min(config.CONFIDENCE_MAX, confidence))
    
    logger.debug(f"Confidence score: {confidence:.1f}% (volatility: {volatility:.2f}, trend: {trend_consistency:.2f})")
    
    current_price = df_model['Close'].iloc[-1]
    avg_move = np.mean(np.abs(price_changes))
    
    scenarios = {
        'best': predicted_price + (avg_move * 1.5),
        'worst': predicted_price - (avg_move * 1.5),
        'likely': (predicted_price * 0.7) + (current_price * 0.3)
    }
    
    logger.debug(f"Prediction complete: ${predicted_price:,.2f} (Confidence: {confidence:.1f}%)")
    return predicted_price, confidence, scenarios

def predict_next_price_v2(df_model, model, scaler, sequence_length=config.SEQUENCE_LENGTH):
    """
    Predict using Model V2 (6 features)
    CRITICAL: Proper inverse transform for 6-column scaler
    Features: Close, RSI, MACD, Signal, ATR, Log Volume
    """
    logger.info(f"Starting LSTM V2 prediction with sequence length: {sequence_length}")
    
    if len(df_model) < sequence_length:
        logger.error(f"Insufficient data for V2: {len(df_model)} rows (need {sequence_length})")
        st.error(f" Data insufficient! Need {sequence_length} rows.")
        return None, None, None
    
    # Take last 60 rows (6 features)
    last_sequence = df_model.iloc[-sequence_length:].values
    last_sequence_scaled = scaler.transform(last_sequence)
    X_input = last_sequence_scaled.reshape(1, sequence_length, 6) # 6 features!
    
    logger.debug(f"V2 Input shape: {X_input.shape}")
    prediction_scaled = model.predict(X_input, verbose=0)
    
    # CRITICAL: Inverse Transform for 6-column scaler
    # Create dummy array with 6 columns
    dummy_array = np.zeros((1, 6))
    # Place predicted value in first column (Close price index)
    dummy_array[:, 0] = prediction_scaled[0, 0]
    # Inverse transform and extract Close price
    predicted_price = scaler.inverse_transform(dummy_array)[:, 0][0]
    
    logger.info(f"V2 Raw prediction: ${predicted_price:,.2f}")
    
    # Confidence Calculation (V2 Enhanced - uses ATR!)
    recent_prices = df_model['Close'].iloc[-10:].values
    price_changes = np.diff(recent_prices)
    volatility = np.std(price_changes)
    trend_consistency = np.abs(np.sum(np.sign(price_changes))) / len(price_changes)
    
    # V2 ENHANCEMENT: Use ATR for more accurate volatility measurement
    current_price = df_model['Close'].iloc[-1]
    current_atr = df_model['ATR_14'].iloc[-1]
    
    # Normalize ATR relative to price (ATR as % of price)
    atr_normalized = current_atr / current_price
    atr_factor = min(atr_normalized * 100, 1.0) # Cap at 1.0
    
    # Calculate base factors
    volatility_factor = min(volatility / np.mean(recent_prices) * 100, 1.0)
    
    # V2 Formula: Base + Trend Bonus - Volatility Penalty - ATR Penalty
    confidence = (config.CONFIDENCE_BASE + 
                  (trend_consistency * config.CONFIDENCE_TREND_WEIGHT) - 
                  (volatility_factor * config.CONFIDENCE_VOLATILITY_WEIGHT) -
                  (atr_factor * config.CONFIDENCE_ATR_WEIGHT)) # NEW: ATR penalty
    
    confidence = max(config.CONFIDENCE_MIN, min(config.CONFIDENCE_MAX, confidence))
    
    logger.info(f"V2 Confidence score: {confidence:.1f}% (volatility: {volatility:.2f}, trend: {trend_consistency:.2f}, ATR: {current_atr:.2f})")

    
    current_price = df_model['Close'].iloc[-1]
    avg_move = np.mean(np.abs(price_changes))
    
    scenarios = {
        'best': predicted_price + (avg_move * 1.5),
        'worst': predicted_price - (avg_move * 1.5),
        'likely': (predicted_price * 0.7) + (current_price * 0.3)
    }
    
    logger.info(f"V2 Prediction complete: ${predicted_price:,.2f} (Confidence: {confidence:.1f}%)")
    return predicted_price, confidence, scenarios


# ==================== COMPARISON PREDICTION (V1 vs V2) ====================
def run_comparison_prediction(df_v1, df_v2, sequence_length=config.SEQUENCE_LENGTH):
    """
    Run both V1 and V2 models simultaneously for comparison.
    Returns predictions, confidences, and scenarios for both models.
    """
    logger.info("Running comparison prediction (V1 vs V2)")
    
    # Run V1 prediction
    pred_v1, conf_v1, scen_v1 = predict_next_price(df_v1, model, scaler, sequence_length)
    
    # Run V2 prediction (if available)
    if model_v2 is not None and scaler_v2 is not None:
        pred_v2, conf_v2, scen_v2 = predict_next_price_v2(df_v2, model_v2, scaler_v2, sequence_length)
    else:
        logger.warning("V2 model not available, using V1 for both")
        pred_v2, conf_v2, scen_v2 = pred_v1, conf_v1, scen_v1
    
    # Calculate differences
    price_diff = pred_v2 - pred_v1 if pred_v1 and pred_v2 else 0
    conf_diff = conf_v2 - conf_v1 if conf_v1 and conf_v2 else 0
    
    comparison_results = {
        'v1': {
            'price': pred_v1,
            'confidence': conf_v1,
            'scenarios': scen_v1
        },
        'v2': {
            'price': pred_v2,
            'confidence': conf_v2,
            'scenarios': scen_v2
        },
        'difference': {
            'price': price_diff,
            'confidence': conf_diff,
            'price_pct': (price_diff / pred_v1 * 100) if pred_v1 else 0
        }
    }
    
    logger.info(f"Comparison complete: V1=${pred_v1:,.2f}, V2=${pred_v2:,.2f}, Diff=${price_diff:,.2f}")
    return comparison_results


# ==================== VISUALISASI PATTERN 60 CANDLE ====================
def create_pattern_chart(df_features, sequence_length=60):
    """
    Visualisasi 60 candle terakhir yang digunakan model untuk prediksi.
    Menampilkan pola yang "dilihat" oleh LSTM.
    """
    # The 60 candles the model "sees" before predicting the next bar.
    pattern_data = df_features.iloc[-sequence_length:].copy()
    x_idx = list(range(len(pattern_data)))

    fig = go.Figure()

    # 1. Main line — muted secondary text color, thin. The data tells the story.
    fig.add_trace(go.Scatter(
        x=x_idx, y=pattern_data['Close'],
        mode='lines',
        name='Close price',
        line=dict(color=TOKENS["text_secondary"], width=1.6),
        hovertemplate='Candle %{x} · <b>$%{y:,.2f}</b><extra></extra>',
    ))

    # 2. Highlight last 10 — accent (clay). The "recent" zone that drives the forecast.
    last_10_x = list(range(len(pattern_data) - 10, len(pattern_data)))
    fig.add_trace(go.Scatter(
        x=last_10_x, y=pattern_data['Close'].iloc[-10:],
        mode='lines',
        name='Most recent 10 candles',
        line=dict(color=TOKENS["accent"], width=2.2),
        hovertemplate='Candle %{x} · <b>$%{y:,.2f}</b><extra></extra>',
    ))

    # 3. Single marker on the current (latest) candle so the "now" point is clear.
    fig.add_trace(go.Scatter(
        x=[x_idx[-1]], y=[pattern_data['Close'].iloc[-1]],
        mode='markers',
        marker=dict(size=8, color=TOKENS["accent"],
                    line=dict(color=TOKENS["surface_base"], width=2)),
        showlegend=False, hoverinfo='skip',
    ))

    fig.update_layout(
        **PLOTLY_THEME,
        height=420,
        xaxis_title="Candle index  ·  0 = oldest, 59 = current",
        yaxis_title="Price (USD)",
    )
    fig.update_xaxes(**PLOTLY_AXIS)
    fig.update_yaxes(**PLOTLY_AXIS, tickprefix="$")
    return fig

# ==================== VISUALISASI CHART UTAMA ====================
def create_main_chart(df, df_features):
    # Slice last 100 candles
    df_viz = df.iloc[-100:]
    df_feat = df_features.iloc[-100:]

    from plotly.subplots import make_subplots

    # 3 stacked panels: price (dominant) → RSI → MACD.
    # Subtitles styled as eyebrow caps (handled by update_annotations below).
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.52, 0.24, 0.24],
        subplot_titles=('PRICE  ·  15M CANDLES', 'RSI 14', 'MACD 12·26·9'),
    )

    # 1. CANDLESTICK — desaturated direction tints, no wick noise.
    fig.add_trace(go.Candlestick(
        x=df_viz.index,
        open=df_viz['Open'], high=df_viz['High'],
        low=df_viz['Low'],  close=df_viz['Close'],
        name='BTC/USDT',
        increasing=dict(line=dict(color=TOKENS["up"],   width=1),
                        fillcolor=TOKENS["up"]),
        decreasing=dict(line=dict(color=TOKENS["down"], width=1),
                        fillcolor=TOKENS["down"]),
        showlegend=False,
    ), row=1, col=1)

    # 2. RSI line — muted secondary, thin. Threshold guides are dashed faint lines.
    fig.add_trace(go.Scatter(
        x=df_feat.index, y=df_feat['RSI_14'],
        name='RSI',
        line=dict(color=TOKENS["text_secondary"], width=1.4),
        hovertemplate='RSI <b>%{y:.1f}</b><extra></extra>',
        showlegend=False,
    ), row=2, col=1)

    fig.add_hline(y=70, line_dash="dot", line_width=1,
                  line_color="rgba(163,145,137,0.45)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_width=1,
                  line_color="rgba(157,165,158,0.45)", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_width=1,
                  line_color="rgba(241,239,236,0.06)", row=2, col=1)

    # 3. MACD — histogram (direction tint), MACD line (accent), signal (faint).
    histogram = df_feat['MACD_12_26_9'] - df_feat['MACDs_12_26_9']
    hist_colors = [TOKENS["up"] if v >= 0 else TOKENS["down"] for v in histogram]
    fig.add_trace(go.Bar(
        x=df_feat.index, y=histogram,
        name='Histogram',
        marker_color=hist_colors,
        opacity=0.55,
        hovertemplate='Histogram <b>%{y:.2f}</b><extra></extra>',
        showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df_feat.index, y=df_feat['MACD_12_26_9'],
        name='MACD',
        line=dict(color=TOKENS["accent"], width=1.6),
        hovertemplate='MACD <b>%{y:.2f}</b><extra></extra>',
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df_feat.index, y=df_feat['MACDs_12_26_9'],
        name='Signal',
        line=dict(color=TOKENS["text_secondary"], width=1.2, dash='dot'),
        hovertemplate='Signal <b>%{y:.2f}</b><extra></extra>',
    ), row=3, col=1)

    # Apply shared axis style to all subplots
    for r in (1, 2, 3):
        fig.update_xaxes(**PLOTLY_AXIS, row=r, col=1)
        fig.update_yaxes(**PLOTLY_AXIS, row=r, col=1)

    # Y-axis formatting per panel
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    # Hide weekend gaps? BTC trades 24/7, so leave x as-is.

    # Subtitle annotations — eyebrow caps aligned left
    fig.update_annotations(
        font=dict(family="JetBrains Mono, monospace", size=10, color=TOKENS["text_muted"]),
        xanchor='left', x=0,
    )

    # Disable Plotly's built-in range slider (visual clutter for our minimal frame)
    fig.update_layout(xaxis_rangeslider_visible=False)

    # Build a layout override that includes a chart-specific legend (overrides theme default)
    main_chart_layout = {
        **PLOTLY_THEME,
        "height": 640,
        "showlegend": True,
        "legend": dict(orientation="h", y=-0.08, x=0.5, xanchor="center",
                       bgcolor="rgba(0,0,0,0)",
                       font=dict(family="JetBrains Mono, monospace",
                                 size=10, color=TOKENS["text_muted"])),
    }
    fig.update_layout(**main_chart_layout)
    return fig

# ==================== MAIN APP LAYOUT ====================
def main():
    # --- Sidebar ---
    with st.sidebar:
        # Compact brand block — small static logo, eyebrow caps, no spin/dance.
        logo_b64 = get_bitcoin_logo_base64()
        st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px;
                        padding:4px 0 24px 0;">
                <img src="data:image/svg+xml;base64,{logo_b64}"
                     alt="Bitcoin"
                     style="width:28px; height:28px; opacity:0.95;" />
                <div style="font-family:'JetBrains Mono', monospace; font-size:11px;
                            letter-spacing:0.18em; text-transform:uppercase;
                            color:var(--qq-text-muted); line-height:1.4;">
                    BTC · LSTM<br/>
                    <span style="color:var(--qq-text-faint);">Predictor</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Model meta — terse, mono labels
        st.markdown(f"""
            <div class="qq-eyebrow" style="margin-bottom:8px;">Model</div>
            <div style="font-family:var(--qq-font-mono); font-size:12px;
                        color:var(--qq-text-secondary); line-height:1.85;">
                <div>LSTM · 60 timesteps</div>
                <div>4 features · Close, RSI, MACD, Signal</div>
                <div>Horizon · 15m</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # Performance section
        st.markdown('<div class="qq-eyebrow">Performance</div>',
                    unsafe_allow_html=True)
        
        # Initialize performance tracking in session state from persistent storage
        if 'performance_tracker' not in st.session_state:
            st.session_state['performance_tracker'] = load_tracker_data()
        
        tracker = st.session_state['performance_tracker']
        
        # Calculate accuracy if we have data
        if len(tracker.get('predictions', [])) > 0:
            metrics = calculate_accuracy_metrics(tracker['predictions'])
            
            # Basic accuracy (fallback if no actual prices yet)
            accuracy = (tracker.get('correct', 0) / len(tracker['predictions'])) * 100 if len(tracker['predictions']) > 0 else 0
            
            # Display performance with enhanced metrics
            if metrics:
                st.metric("Directional accuracy", f"{metrics['directional_accuracy']:.1f}%")
                st.caption(f"MAE  ${metrics['mae']:,.2f}")
                st.caption(f"RMSE ${metrics['rmse']:,.2f}")
                st.caption(f"{metrics['total_predictions']} verified")
            else:
                st.metric("Accuracy", f"{accuracy:.1f}%")
                st.caption(f"{len(tracker['predictions'])} predictions")
                st.caption("Awaiting actual prices")

            # Reset button
            if st.button("Reset tracker", use_container_width=True):
                # Delete JSON file
                if os.path.exists(TRACKER_FILE):
                    os.remove(TRACKER_FILE)
                    logger.info("Tracker data file deleted")

                # Reset session state
                st.session_state['performance_tracker'] = {
                    'predictions': [],
                    'correct': 0,
                    'last_actual_price': None
                }
                st.success("Tracker cleared.")
                st.rerun()

            # Manual Update Actual Prices Button
            st.write("---")
            if st.button("Update actual prices", use_container_width=True, help="Manually update all predictions with current price for testing"):
                st.session_state['trigger_manual_update'] = True
                st.rerun()

        else:
            # Elegant empty state — explains what triggers the metric
            st.markdown("""
                <div style="padding:14px 16px; border:1px dashed var(--qq-border-default);
                            border-radius:var(--qq-radius-md);
                            color:var(--qq-text-muted); font-size:12.5px; line-height:1.55;">
                    Run a prediction to start tracking.<br/>
                    Each forecast is matched against the actual close after 15 minutes.
                </div>
            """, unsafe_allow_html=True)

        # Prediction History & Error Analysis
        if len(tracker.get('predictions', [])) > 0:
            st.write("---")
            with st.expander("History · last 10 predictions", expanded=False):
                st.caption("Individual errors with outlier flag (>2σ)")
                
                pred_data = []
                for i, pred in enumerate(tracker['predictions'][-10:], 1): # Last 10
                    if pred.get('actual_price') is not None:
                        error = abs(pred['predicted_price'] - pred['actual_price'])
                        direction_correct = "Benar" if pred.get('direction') == ('up' if pred['actual_price'] > pred['current_price'] else 'down') else "Salah"
                        pred_data.append({
                            '#': i,
                            'Predicted': f"${pred['predicted_price']:.2f}",
                            'Actual': f"${pred['actual_price']:.2f}",
                            'Error': f"${error:.2f}",
                            'Dir': direction_correct
                        })
                
                if pred_data:
                    import pandas as pd
                    df_hist = pd.DataFrame(pred_data)
                    st.dataframe(df_hist, use_container_width=True, hide_index=True)
                    
                    # Highlight outliers
                    errors = [float(d['Error'].replace('$','').replace(',','')) for d in pred_data]
                    if len(errors) > 2:
                        mean_error = np.mean(errors)
                        std_error = np.std(errors)
                        outliers = [i+1 for i, e in enumerate(errors) if abs(e - mean_error) > 2 * std_error]
                        if outliers:
                            st.warning(f"Outliers flagged: #{', #'.join(map(str, outliers))}")
                else:
                    st.caption("No verified predictions yet")

        st.write("---")
        st.markdown('<div class="qq-eyebrow">Tools</div>', unsafe_allow_html=True)

        # Backtesting Section
        with st.expander("Backtest on historical range", expanded=False):
            st.caption("Replay the model over past 15-minute candles")
            
            # Date range selector
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=7),
                    max_value=datetime.now() - timedelta(days=1)
                )
            with col_date2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now() - timedelta(days=1),
                    max_value=datetime.now()
                )
            
            # Run backtest button
            if st.button("Run backtest", use_container_width=True):
                if start_date >= end_date:
                    st.error("Start date must be before end date.")
                else:
                    with st.spinner("Running backtest…"):
                        metrics, error = run_backtest(
                            start_date, end_date,
                            model, scaler
                        )

                        if error:
                            st.error(f"Backtest failed: {error}")
                        elif metrics:
                            st.session_state['backtest_results'] = metrics
                            st.success("Backtest complete.")
                            st.rerun()
                        else:
                            st.error("No results generated.")

        # Force Refresh Button
        if st.button("Force refresh data", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared. Reloading fresh data…")
            st.rerun()

        # Telegram Alert Settings
        st.write("---")
        st.markdown('<div class="qq-eyebrow">Alerts</div>', unsafe_allow_html=True)

        # Bot Credentials
        with st.expander("Telegram credentials", expanded=False):
            bot_token = st.text_input(
                "Bot token",
                value="",
                type="password",
                help="Get from @BotFather on Telegram",
                key="telegram_bot_token"
            )
            chat_id = st.text_input(
                "Chat ID",
                value="",
                help="Get from @userinfobot on Telegram",
                key="telegram_chat_id"
            )

            # Test Connection Button
            if st.button("Test connection", use_container_width=True):
                if bot_token and chat_id:
                    test_msg = f"""
<b>Connection test successful</b>

Bot is connected to your Telegram account.
You will receive alerts here.

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
                    success, msg = send_telegram_message(bot_token, chat_id, test_msg)
                    if success:
                        st.success("Test message sent. Check your Telegram.")
                    else:
                        st.error(f"Failed: {msg}")
                else:
                    st.warning("Please enter both Bot Token and Chat ID.")

        # Alert Toggles
        st.markdown('<div class="qq-eyebrow" style="margin-top:14px;">Triggers</div>',
                    unsafe_allow_html=True)
        alert_rsi_overbought = st.checkbox("RSI overbought (>70)", value=False, key="alert_rsi_ob")
        alert_rsi_oversold   = st.checkbox("RSI oversold (<30)",   value=False, key="alert_rsi_os")
        alert_macd           = st.checkbox("MACD signal cross",    value=False, key="alert_macd")
        alert_prediction     = st.checkbox("Prediction result",    value=False, key="alert_pred")
        
        # Store alert settings in session state
        if 'alert_settings' not in st.session_state:
            st.session_state['alert_settings'] = {}
        
        st.session_state['alert_settings'] = {
            'rsi_overbought': alert_rsi_overbought,
            'rsi_oversold': alert_rsi_oversold,
            'macd_crossover': alert_macd,
            'prediction': alert_prediction
        }
        
        # Note: bot_token and chat_id are automatically stored in session_state
        # by Streamlit widgets with key="telegram_bot_token" and key="telegram_chat_id"
        # No need to manually assign them here
        
        # Auto-Refresh Toggle
        st.markdown('<div class="qq-eyebrow" style="margin-top:14px;">Mode</div>',
                    unsafe_allow_html=True)
        refresh = st.checkbox("Auto-refresh (live)", value=False,
                              help="Re-run the app on every interaction tick.")
        if refresh:
            st.rerun()

        # Author Credit — quiet, no gradients, mono labels
        st.write("---")
        st.markdown(f"""
            <div style="padding:14px 0 4px 0; line-height:1.6;">
                <div class="qq-eyebrow" style="margin-bottom:8px;">Author</div>
                <div style="font-family:var(--qq-font-sans); font-size:13.5px;
                            font-weight:600; color:var(--qq-text-primary);">
                    {config.AUTHOR_NAME}
                </div>
                <div style="font-family:var(--qq-font-mono); font-size:11.5px;
                            color:var(--qq-text-muted);">
                    {config.AUTHOR_NIM} · {config.AUTHOR_PROGRAM}
                </div>
                <div style="font-family:var(--qq-font-mono); font-size:11px;
                            color:var(--qq-text-faint); margin-top:4px;">
                    Skripsi · {config.COPYRIGHT_YEAR}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ===== DATA LOAD ===== (moved above the header so we have authoritative timestamps)
    # Split into two spinners so the user can SEE which phase is slow
    # (network vs. indicator computation). Helps diagnose hangs.
    _t0 = datetime.now()
    with st.spinner("Fetching market data from Binance…"):
        df_raw = get_live_bitcoin_data()
    _t1 = datetime.now()
    with st.spinner("Computing technical indicators…"):
        df_full, df_model = calculate_technical_indicators(df_raw)
    _t2 = datetime.now()
    logger.info(
        f"Render phase timing — fetch: {(_t1 - _t0).total_seconds():.2f}s, "
        f"indicators: {(_t2 - _t1).total_seconds():.2f}s"
    )

    # Auto-update actual prices for predictions older than 15 minutes
    if 'performance_tracker' in st.session_state:
        tracker = st.session_state['performance_tracker']
        updated_count = update_actual_prices(tracker, df_raw)
        if updated_count > 0:
            save_tracker_data(tracker)
            logger.info(f"Auto-updated {updated_count} predictions with actual prices")

    # Handle manual update trigger
    if st.session_state.get('trigger_manual_update', False):
        if 'performance_tracker' in st.session_state:
            tracker = st.session_state['performance_tracker']
            _current_price = df_raw['Close'].iloc[-1]
            updated = manual_update_all_actual_prices(tracker, _current_price)
            if updated > 0:
                save_tracker_data(tracker)
                st.session_state['performance_tracker'] = tracker
                st.success(f"Updated {updated} predictions with actual prices.")
                st.info("Refresh to see updated metrics.")
            else:
                st.info("All predictions already have actual prices.")
        st.session_state['trigger_manual_update'] = False

    # ===== HEADER ===== (Quiet Quant: hero strip + page header)
    current_price = df_raw['Close'].iloc[-1]
    prev_price    = df_raw['Close'].iloc[-2]
    delta         = current_price - prev_price
    delta_pct     = (delta / prev_price) * 100 if prev_price else 0.0

    last_candle_time = df_raw.index[-1]
    next_candle_time = last_candle_time + timedelta(minutes=15)
    data_source_used = st.session_state.get('data_source_used', 'unknown')

    # 1. HERO STRIP — small logo + status row
    logo_b64 = get_bitcoin_logo_base64()
    source_label = "BINANCE · BTCUSDT" if data_source_used == "binance" else "YFINANCE · BTC-USD"
    status_html = (
        styles.live_dot("LIVE")
        + '<span class="qq-divider"></span>'
        + f'<span>{source_label}</span>'
        + '<span class="qq-divider"></span>'
        + f'<span>{last_candle_time.strftime("%Y-%m-%d · %H:%M UTC")}</span>'
    )
    st.markdown(
        styles.hero_strip(
            brand_label="BTC · LSTM PREDICTOR · 15M",
            logo_b64=logo_b64,
            status_html=status_html,
        ),
        unsafe_allow_html=True,
    )

    # 2. PAGE HEADER — h1 + subtitle + meta (timestamps)
    meta_html = (
        f'<div>Last candle&nbsp;&nbsp;{last_candle_time.strftime("%H:%M")} UTC</div>'
        f'<div>Next forecast&nbsp;&nbsp;{next_candle_time.strftime("%H:%M")} UTC</div>'
    )
    st.markdown(
        styles.page_header(
            eyebrow_text="Bitcoin · 15-minute intraday",
            title="Intraday price forecast.",
            subtitle="A long short-term memory network conditioned on RSI and MACD features, "
                     "trained to project the next 15-minute closing price.",
            meta_html=meta_html,
        ),
        unsafe_allow_html=True,
    )

    # ===== METRICS ROW =====
    # Custom HTML cards so the primary one can carry the accent border-left.
    rsi_val    = df_full['RSI_14'].iloc[-1]
    macd_val   = df_full['MACD_12_26_9'].iloc[-1]
    signal_val = df_full['MACDs_12_26_9'].iloc[-1]
    hist       = macd_val - signal_val

    if   rsi_val > 70: rsi_label, rsi_variant = "Overbought", "down"
    elif rsi_val < 30: rsi_label, rsi_variant = "Oversold",   "up"
    else:              rsi_label, rsi_variant = "Neutral",    "neutral"

    macd_label   = "Bullish" if hist > 0 else "Bearish"
    macd_variant = "up"      if hist > 0 else "down"

    delta_variant = "up" if delta >= 0 else "down"
    delta_icon    = ICONS["arrow_up"] if delta >= 0 else ICONS["arrow_down"]

    def _metric_card_html(eyebrow, value, sub_html, primary=False):
        return f"""
            <div class="{'qq-card-primary' if primary else 'qq-card'}"
                 style="height:100%;">
              <div class="qq-eyebrow">{eyebrow}</div>
              <div class="qq-metric-hero" style="font-size:34px; margin:10px 0 8px 0;">{value}</div>
              <div>{sub_html}</div>
            </div>
        """

    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        sub = (
            f'<span class="qq-pill qq-pill-{delta_variant}">'
            f'  <span style="display:inline-flex">{delta_icon}</span>'
            f'  {delta:+,.2f}  ·  {delta_pct:+.2f}%'
            f'</span>'
        )
        st.markdown(_metric_card_html(
            "BITCOIN  ·  USD",
            f"${current_price:,.2f}",
            sub,
            primary=True,
        ), unsafe_allow_html=True)

    with m2:
        sub = (
            f'<span class="qq-pill qq-pill-{rsi_variant}">'
            f'  {rsi_label}'
            f'</span>'
            f'<span style="margin-left:8px; font-family:var(--qq-font-mono); '
            f'font-size:11px; color:var(--qq-text-faint);">14-period</span>'
        )
        st.markdown(_metric_card_html("RSI", f"{rsi_val:.1f}", sub), unsafe_allow_html=True)

    with m3:
        hist_icon = ICONS["arrow_up"] if hist >= 0 else ICONS["arrow_down"]
        sub = (
            f'<span class="qq-pill qq-pill-{macd_variant}">'
            f'  <span style="display:inline-flex">{hist_icon}</span>'
            f'  {macd_label}'
            f'</span>'
            f'<span style="margin-left:8px; font-family:var(--qq-font-mono); '
            f'font-size:11px; color:var(--qq-text-faint);">hist {hist:+.2f}</span>'
        )
        st.markdown(_metric_card_html("MACD  ·  12·26·9", f"{macd_val:.2f}", sub),
                    unsafe_allow_html=True)
    
    # Check and send Telegram alerts (if enabled)
    if 'alert_settings' in st.session_state and st.session_state.get('alert_settings'):
        bot_token = st.session_state.get('telegram_bot_token', '')
        chat_id = st.session_state.get('telegram_chat_id', '')
        
        if bot_token and chat_id:
            alerts_sent = check_and_send_alerts(
                bot_token, 
                chat_id, 
                rsi_val, 
                macd_val, 
                signal_val, 
                current_price,
                st.session_state['alert_settings']
            )
            
            if alerts_sent:
                logger.info(f"Alerts sent: {', '.join(alerts_sent)}")
    
    
    # (Timestamps and data source live in the hero strip + page header already.)

    # ===== MARKET ACTIVITY =====
    st.markdown(
        styles.section_header("Market activity", "PAST 100 CANDLES · 25h"),
        unsafe_allow_html=True,
    )
    st.plotly_chart(create_main_chart(df_raw, df_full), use_container_width=True,
                    config={"displayModeBar": False})

    # TradingView Interactive Chart — LAZY MOUNT.
    # st.expander still executes its body when collapsed, so previously the iframe was
    # loading on every render. Now we gate the iframe behind an explicit toggle so the
    # initial page render never waits on tradingview.com's CDN.
    with st.expander("Open interactive TradingView chart", expanded=False):
        st.caption(
            "Loads the full TradingView widget (trendlines, Fibonacci, drawing tools). "
            "Pulled from tradingview.com — may take a few seconds the first time."
        )
        if st.button("Load TradingView widget", key="load_tv_widget"):
            st.session_state["tv_widget_loaded"] = True

        if st.session_state.get("tv_widget_loaded", False):
            render_tradingview_widget()
        else:
            st.markdown(
                '<div style="padding:24px; border:1px dashed var(--qq-border-default);'
                'border-radius:var(--qq-radius-md); color:var(--qq-text-muted);'
                'font-size:13px; text-align:center;">'
                'Click <strong style="color:var(--qq-text-primary);">Load TradingView widget</strong> '
                'above to mount the iframe on demand.'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("""
            <a href="https://www.tradingview.com/chart/?symbol=BINANCE:BTCUSDT"
               target="_blank" rel="noopener"
               style="display:inline-flex; align-items:center; gap:8px;
                      padding:8px 14px; border-radius:var(--qq-radius-sm);
                      background:var(--qq-surface-overlay);
                      border:1px solid var(--qq-border-default);
                      color:var(--qq-text-primary);
                      font-family:var(--qq-font-sans); font-size:12.5px; font-weight:500;
                      text-decoration:none; margin-top:10px;">
                Or open in TradingView (full app) →
            </a>
        """, unsafe_allow_html=True)


    # ===== LSTM FORECAST =====
    st.markdown(
        styles.section_header("LSTM forecast",
                              f"+15 MIN HORIZON · {next_candle_time.strftime('%H:%M UTC')}"),
        unsafe_allow_html=True,
    )

    # Prediction CTA — primary button on the left, terse caption on the right
    cta_col, cta_hint = st.columns([1, 3], gap="medium")
    with cta_col:
        run_prediction = st.button("Run prediction", use_container_width=True, type="primary")
    with cta_hint:
        st.markdown(
            f'<div style="padding-top:9px; font-family:var(--qq-font-mono); font-size:12px; '
            f'color:var(--qq-text-muted);">'
            f'Projects the <span style="color:var(--qq-text-primary);">'
            f'{next_candle_time.strftime("%H:%M UTC")}</span> close from the last 60 candles · '
            f'inference ≈ 1s'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Prediction action handler — logic preserved verbatim from original.
    if run_prediction:
        is_valid, validation_msg = validate_data_for_prediction(df_raw, min_rows=60)

        if not is_valid:
            st.error(validation_msg)
            st.warning("Try **Force refresh data** in the sidebar to pull the latest candles.")
        else:
            try:
                with st.spinner("Running LSTM inference…"):
                    pred_price, conf, scenarios = predict_next_price(df_model, model, scaler)

                    if pred_price:
                        diff = pred_price - current_price
                        pct_diff = (diff / current_price) * 100

                        st.session_state['last_pred'] = {
                            'price': pred_price, 'conf': conf, 'scenarios': scenarios,
                            'diff': diff, 'pct': pct_diff
                        }

                        # Track for performance monitoring
                        tracker = st.session_state.get('performance_tracker', {
                            'predictions': [], 'correct': 0, 'last_actual_price': None
                        })
                        tracker['predictions'].append({
                            'timestamp':       datetime.now(),
                            'predicted_price': pred_price,
                            'current_price':   current_price,
                            'direction':       'up' if diff > 0 else 'down',
                        })
                        st.session_state['performance_tracker'] = tracker
                        save_tracker_data(tracker)
                        logger.info(f"Prediction tracked: ${pred_price:,.2f}")
                        logger.info(f"Tracker state: {len(tracker['predictions'])} predictions")

                        # Telegram alert for prediction (if enabled)
                        if st.session_state.get('alert_settings', {}).get('prediction', False):
                            bot_token = st.session_state.get('telegram_bot_token', '')
                            chat_id   = st.session_state.get('telegram_chat_id', '')
                            if bot_token and chat_id:
                                direction  = "NAIK" if diff > 0 else "TURUN"
                                conf_level = ("TINGGI" if conf >= 70
                                              else "SEDANG" if conf >= 55
                                              else "RENDAH")
                                pred_time  = (datetime.now() + timedelta(minutes=15)).strftime('%H:%M')

                                pred_msg = f"""
<b>LSTM PREDICTION ALERT</b>

Current Price: ${current_price:,.2f}
Predicted Price: ${pred_price:,.2f}

Change: {direction} {abs(pct_diff):.2f}%
Difference: ${abs(diff):,.2f}

Confidence: {conf_level} ({conf:.1f}%)

Scenarios:
  • Best: ${scenarios['best']:,.2f}
  • Likely: ${scenarios['likely']:,.2f}
  • Worst: ${scenarios['worst']:,.2f}

Prediction Time: {pred_time} UTC

Disclaimer: For reference only, not financial advice.
"""
                                send_telegram_message(bot_token, chat_id, pred_msg)
                    else:
                        st.error("Model failed to produce a prediction. Please try again.")

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.warning(
                    "**Troubleshooting**\n"
                    "1. Confirm `model_bitcoin_v1_4features.keras` is present.\n"
                    "2. Confirm `scaler_bitcoin_v1.pkl` is present.\n"
                    "3. Use **Force refresh data** in the sidebar."
                )
                import traceback
                with st.expander("Stack trace"):
                    st.code(traceback.format_exc())

    # ----- PREDICTION RESULT PANEL (60/40) -----
    if 'last_pred' in st.session_state:
        res = st.session_state['last_pred']
        is_up = res['diff'] >= 0
        dir_variant = "up" if is_up else "down"
        dir_icon    = ICONS["arrow_up"] if is_up else ICONS["arrow_down"]

        if   res['conf'] >= 70: conf_band = ("High",   "high")
        elif res['conf'] >= 55: conf_band = ("Medium", "mid")
        else:                   conf_band = ("Low",    "low")

        pred_time_str = (datetime.now() + timedelta(minutes=15)).strftime('%H:%M')

        left_col, right_col = st.columns([3, 2], gap="large")

        # --- LEFT 60% — PRIMARY PREDICTION CARD ---
        with left_col:
            primary_html = f"""
                <div class="qq-card-primary" style="height:100%;">
                    <div class="qq-eyebrow">PREDICTED CLOSE · {pred_time_str} UTC</div>
                    <div class="qq-metric-hero">${res['price']:,.2f}</div>
                    <div class="qq-arrow-row">
                        <span>From <span class="qq-value-to">${current_price:,.2f}</span></span>
                        <span class="qq-arrow">{ICONS["arrow_right"]}</span>
                        <span>To <span class="qq-value-to">${res['price']:,.2f}</span></span>
                    </div>
                    <div style="margin-top:18px;">
                        <span class="qq-pill qq-pill-{dir_variant}">
                            <span style="display:inline-flex">{dir_icon}</span>
                            {res['diff']:+,.2f}  ·  {res['pct']:+.2f}%
                        </span>
                    </div>
                </div>
            """
            st.markdown(primary_html, unsafe_allow_html=True)

        # --- RIGHT 40% — CONFIDENCE + SCENARIOS ---
        with right_col:
            scen = res['scenarios']
            right_html = f"""
                <div class="qq-card" style="height:100%;">
                    <div class="qq-eyebrow">CONFIDENCE</div>
                    <div style="display:flex; align-items:baseline; gap:10px; margin-top:6px;">
                        <div class="qq-mono"
                             style="font-size:28px; font-weight:600;
                                    color:var(--qq-text-primary);
                                    letter-spacing:-0.02em;">
                            {res['conf']:.1f}%
                        </div>
                        <span class="qq-pill qq-pill-neutral">{conf_band[0]}</span>
                    </div>
                    {styles.meter_bar(res['conf'], conf_band[1])}
                    <div style="margin-top:18px; padding-top:14px;
                                border-top:1px solid var(--qq-border-subtle);">
                        <div class="qq-eyebrow" style="margin-bottom:8px;">Scenarios</div>
                        <div class="qq-scenario">
                            <span class="qq-label">Bull</span>
                            <span class="qq-value">${scen['best']:,.2f}</span>
                        </div>
                        <div class="qq-scenario">
                            <span class="qq-label">Base</span>
                            <span class="qq-value">${scen['likely']:,.2f}</span>
                        </div>
                        <div class="qq-scenario">
                            <span class="qq-label">Bear</span>
                            <span class="qq-value">${scen['worst']:,.2f}</span>
                        </div>
                    </div>
                </div>
            """
            st.markdown(right_html, unsafe_allow_html=True)

        # ----- INTERPRETATION + DISCLAIMER -----
        direction_text = "rise" if is_up else "fall"
        st.markdown(f"""
            <div class="qq-card" style="margin-top:24px;">
                <div class="qq-eyebrow" style="margin-bottom:10px;">Reading</div>
                <div style="font-family:var(--qq-font-sans); font-size:14.5px;
                            color:var(--qq-text-secondary); line-height:1.65;">
                    Based on the last 60 candles of Close, RSI, MACD and Signal, the model
                    expects price to <strong style="color:var(--qq-text-primary);">{direction_text} {abs(res['pct']):.2f}%</strong>
                    over the next 15 minutes, landing near
                    <strong style="color:var(--qq-text-primary);">${res['price']:,.2f}</strong> at {pred_time_str} UTC.
                    Internal confidence is <strong style="color:var(--qq-text-primary);">{conf_band[0].lower()}</strong>
                    ({res['conf']:.1f}%).
                </div>
            </div>

            <div style="margin-top:14px; padding:14px 18px;
                        border:1px solid var(--qq-border-subtle);
                        border-left:2px solid var(--qq-warning);
                        border-radius:var(--qq-radius-md);
                        background:var(--qq-warning-soft);
                        font-family:var(--qq-font-sans); font-size:12.5px;
                        color:var(--qq-text-secondary); line-height:1.6;">
                <strong style="color:var(--qq-text-primary);">Not financial advice.</strong>
                A 15-minute LSTM has high variance — historical directional accuracy sits in
                the 55–65% band. The model is blind to news, macro events, and order-book flow.
                Treat as one signal among many.
            </div>
        """, unsafe_allow_html=True)

        # ----- SEQUENCE INPUT (60-candle pattern) -----
        st.markdown(
            styles.section_header("Sequence input",
                                  "LAST 60 CANDLES · MODEL'S RECEPTIVE FIELD"),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="margin-top:-12px; margin-bottom:18px; color:var(--qq-text-muted); '
            'font-size:13px; line-height:1.55; max-width:70ch;">'
            'The 60 most recent 15-minute candles the LSTM sees before producing its '
            'forecast. The accent-coloured tail (last 10) carries the most weight in the '
            'sequence representation.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(create_pattern_chart(df_model), use_container_width=True,
                        config={"displayModeBar": False})

    # ===== ACCURACY TREND =====
    if 'performance_tracker' in st.session_state:
        tracker_for_trend = st.session_state['performance_tracker']
        verified_count = len([p for p in tracker_for_trend.get('predictions', [])
                              if p.get('actual_price') is not None])

        st.markdown(
            styles.section_header("Accuracy trend",
                                  f"{verified_count} VERIFIED · CUMULATIVE"),
            unsafe_allow_html=True,
        )

        trend_chart = create_accuracy_trend_chart(tracker_for_trend)
        if trend_chart:
            st.plotly_chart(trend_chart, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            # Elegant empty state
            st.markdown("""
                <div style="padding:32px 28px; border:1px dashed var(--qq-border-default);
                            border-radius:var(--qq-radius-md); text-align:center;
                            color:var(--qq-text-muted);">
                    <div class="qq-eyebrow" style="margin-bottom:8px;">No data yet</div>
                    <div style="font-family:var(--qq-font-sans); font-size:13.5px;
                                color:var(--qq-text-secondary); max-width:46ch; margin:0 auto;
                                line-height:1.55;">
                        Run at least two predictions and wait 15 minutes for the actual
                        candles to land. The chart will start populating then.
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # ===== BACKTEST RESULTS =====
    if 'backtest_results' in st.session_state:
        bt_results = st.session_state['backtest_results']
        if bt_results.get('v1'):
            v1 = bt_results['v1']

            st.markdown(
                styles.section_header("Backtest",
                                      f"{v1['total_predictions']:,} HISTORICAL CANDLES"),
                unsafe_allow_html=True,
            )

            # 4-stat panel: directional · MAE · RMSE · sample
            # Directional uses a progress bar so the headline number breathes.
            st.markdown(f"""
                <div class="qq-card-primary" style="padding:24px 28px;">
                    <div class="qq-eyebrow">DIRECTIONAL ACCURACY</div>
                    <div style="display:flex; align-items:baseline; gap:12px; margin:8px 0 4px 0;">
                        <div class="qq-mono"
                             style="font-size:42px; font-weight:600;
                                    color:var(--qq-text-primary);
                                    letter-spacing:-0.02em;">
                            {v1['directional_accuracy']:.2f}%
                        </div>
                        <span class="qq-pill qq-pill-neutral">{v1['total_predictions']:,} samples</span>
                    </div>
                    {styles.meter_bar(v1['directional_accuracy'])}
                </div>

                <div style="display:grid;
                            grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));
                            gap:16px; margin-top:16px;">
                    <div class="qq-card">
                        <div class="qq-eyebrow">MAE</div>
                        <div class="qq-mono" style="font-size:22px; font-weight:600;
                                                    color:var(--qq-text-primary); margin-top:8px;
                                                    letter-spacing:-0.01em;">
                            ${v1['mae']:,.2f}
                        </div>
                    </div>
                    <div class="qq-card">
                        <div class="qq-eyebrow">RMSE</div>
                        <div class="qq-mono" style="font-size:22px; font-weight:600;
                                                    color:var(--qq-text-primary); margin-top:8px;
                                                    letter-spacing:-0.01em;">
                            ${v1['rmse']:,.2f}
                        </div>
                    </div>
                    <div class="qq-card">
                        <div class="qq-eyebrow">MEDIAN ERROR</div>
                        <div class="qq-mono" style="font-size:22px; font-weight:600;
                                                    color:var(--qq-text-primary); margin-top:8px;
                                                    letter-spacing:-0.01em;">
                            ${v1['median_error']:,.2f}
                        </div>
                    </div>
                    <div class="qq-card">
                        <div class="qq-eyebrow">ERROR RANGE</div>
                        <div class="qq-mono" style="font-size:14px; font-weight:500;
                                                    color:var(--qq-text-secondary); margin-top:10px;
                                                    line-height:1.5;">
                            min ${v1['min_error']:,.2f}<br/>
                            max ${v1['max_error']:,.2f}
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # ===== FOOTER =====
    st.markdown(f"""
        <div style="margin-top:80px; padding:32px 0 24px 0;
                    border-top:1px solid var(--qq-border-subtle);">
            <div style="display:flex; flex-wrap:wrap; gap:32px;
                        justify-content:space-between; align-items:flex-start;">
                <div style="max-width:48ch;">
                    <div class="qq-eyebrow" style="margin-bottom:10px;">Project</div>
                    <div style="font-family:var(--qq-font-sans); font-size:15px;
                                font-weight:600; color:var(--qq-text-primary);
                                line-height:1.4;">
                        Bitcoin intraday forecasting with LSTM, RSI and MACD
                    </div>
                    <div style="font-family:var(--qq-font-sans); font-size:13px;
                                color:var(--qq-text-muted); margin-top:6px; line-height:1.55;">
                        {config.APP_SUBTITLE}
                    </div>
                </div>
                <div>
                    <div class="qq-eyebrow" style="margin-bottom:10px;">Author</div>
                    <div style="font-family:var(--qq-font-sans); font-size:14px;
                                font-weight:600; color:var(--qq-text-primary);">
                        {config.AUTHOR_NAME}
                    </div>
                    <div style="font-family:var(--qq-font-mono); font-size:12px;
                                color:var(--qq-text-muted); margin-top:4px;">
                        {config.AUTHOR_NIM} · {config.AUTHOR_PROGRAM}
                    </div>
                </div>
                <div>
                    <div class="qq-eyebrow" style="margin-bottom:10px;">Stack</div>
                    <div style="font-family:var(--qq-font-mono); font-size:12px;
                                color:var(--qq-text-secondary); line-height:1.7;">
                        Streamlit · TensorFlow<br/>
                        Plotly · Binance API
                    </div>
                </div>
            </div>
            <div style="margin-top:32px; padding-top:18px;
                        border-top:1px solid var(--qq-border-subtle);
                        display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px;
                        font-family:var(--qq-font-mono); font-size:11px;
                        color:var(--qq-text-faint); letter-spacing:0.04em;">
                <span>© {config.COPYRIGHT_YEAR} · {config.AUTHOR_NAME}</span>
                <span>Skripsi · {config.AUTHOR_PROGRAM}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
