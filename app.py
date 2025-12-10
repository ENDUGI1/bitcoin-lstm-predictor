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

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)
logger.info("🚀 Bitcoin LSTM Dashboard started")

# ==================== KONFIGURASI PAGE ====================
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="bitcoin-btc-logo.png",  # Custom Bitcoin logo
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ... (CSS Injection remains checks out, skip to Feature Engineering)

# ==================== MANUAL INDICATOR FUNCTIONS (NO DEPENDENCIES) ====================
def calculate_rsi(series, period=14):
    """Calculate RSI manually using Pandas"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
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

def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR) manually
    ATR = Moving Average of True Range
    True Range = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    
    return atr

# ==================== CUSTOM CSS (CYBERPUNK STYLE) ====================
def inject_custom_css():
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

        /* Base Typography */
        html, body, [class*="css"] {
            font-family: 'Exo 2', sans-serif !important;
            color: #E0E0E0;
        }

        /* Titles and Headers */
        h1, h2, h3 {
            font-weight: 700;
            background: linear-gradient(90deg, #00D9FF, #BD00FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
        }
        
        .metric-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #888;
            text-transform: uppercase;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #FFF;
        }

        /* Cards/Containers */
        div[data-testid="stMetric"], div[data-testid="stExpander"] {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }

        div[data-testid="stMetric"]:hover {
            border-color: #00D9FF;
            box-shadow: 0 0 15px rgba(0, 217, 255, 0.2);
        }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 1px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 114, 255, 0.4);
        }

        div.stButton > button:active {
            transform: translateY(0);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #080A10;
            border-right: 1px solid rgba(255,255,255,0.1);
        }

        /* Plotly Chart Background */
        .js-plotly-plot .plotly .main-svg {
            background: transparent !important;
        }
        
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ==================== PERSISTENT TRACKER STORAGE ====================
TRACKER_FILE = "tracker_data.json"

def load_tracker_data():
    """Load tracker data from JSON file"""
    try:
        if os.path.exists(TRACKER_FILE):
            with open(TRACKER_FILE, 'r') as f:
                data = json.load(f)
                logger.info(f"Tracker data loaded: V1={len(data.get('v1_predictions', []))}, V2={len(data.get('v2_predictions', []))}")
                return data
        else:
            logger.info("No existing tracker data found, starting fresh")
            return {
                'v1_predictions': [],
                'v2_predictions': [],
                'v1_correct': 0,
                'v2_correct': 0,
                'last_actual_price': None
            }
    except Exception as e:
        logger.error(f"Error loading tracker data: {str(e)}, starting fresh")
        return {
            'v1_predictions': [],
            'v2_predictions': [],
            'v1_correct': 0,
            'v2_correct': 0,
            'last_actual_price': None
        }

def save_tracker_data(tracker_data):
    """Save tracker data to JSON file"""
    try:
        # Convert datetime objects to strings for JSON serialization
        data_to_save = {
            'v1_predictions': [],
            'v2_predictions': [],
            'v1_correct': tracker_data.get('v1_correct', 0),
            'v2_correct': tracker_data.get('v2_correct', 0),
            'last_actual_price': tracker_data.get('last_actual_price')
        }
        
        # Convert predictions with datetime to serializable format
        for pred in tracker_data.get('v1_predictions', []):
            pred_copy = pred.copy()
            if isinstance(pred_copy.get('timestamp'), datetime):
                pred_copy['timestamp'] = pred_copy['timestamp'].isoformat()
            data_to_save['v1_predictions'].append(pred_copy)
        
        for pred in tracker_data.get('v2_predictions', []):
            pred_copy = pred.copy()
            if isinstance(pred_copy.get('timestamp'), datetime):
                pred_copy['timestamp'] = pred_copy['timestamp'].isoformat()
            data_to_save['v2_predictions'].append(pred_copy)
        
        # Keep only last 100 predictions to avoid file bloat
        data_to_save['v1_predictions'] = data_to_save['v1_predictions'][-100:]
        data_to_save['v2_predictions'] = data_to_save['v2_predictions'][-100:]
        
        with open(TRACKER_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        logger.info(f"Tracker data saved: V1={len(data_to_save['v1_predictions'])}, V2={len(data_to_save['v2_predictions'])}")
        return True
    except Exception as e:
        logger.error(f"Error saving tracker data: {str(e)}")
        return False

# ==================== LOAD MODEL & SCALER ====================
@st.cache_resource
def load_model_and_scaler():
    """Load pretrained LSTM model V1 and scaler (4 features)"""
    try:
        model = tf.keras.models.load_model(config.MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        logger.info("✅ Model V1 (4 features) loaded successfully")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading Model V1: {str(e)}")
        st.stop()

@st.cache_resource
def load_model_v2():
    """Load pretrained LSTM model V2 and scaler (6 features)"""
    try:
        model_v2 = tf.keras.models.load_model(config.MODEL_V2_PATH)
        scaler_v2 = joblib.load(config.SCALER_V2_PATH)
        logger.info("✅ Model V2 (6 features) loaded successfully")
        return model_v2, scaler_v2
    except FileNotFoundError as e:
        logger.warning(f"Model V2 files not found: {str(e)}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading Model V2: {str(e)}")
        return None, None

# Load both models
model, scaler = load_model_and_scaler()
model_v2, scaler_v2 = load_model_v2()

# ==================== FUNGSI AMBIL DATA LIVE ====================
@st.cache_data(ttl=config.CACHE_TTL_DATA, show_spinner=False)
def get_live_bitcoin_data():
    """Fetch live data from Yahoo Finance with Retry Logic"""
    import time
    logger.info(f"Fetching Bitcoin data: {config.TICKER_SYMBOL}, Period: {config.DATA_PERIOD}, Interval: {config.DATA_INTERVAL}")
    
    for i in range(config.MAX_RETRIES):
        try:
            # Try fetching with yf.download wrapper which proved more stable
            df = yf.download(config.TICKER_SYMBOL, period=config.DATA_PERIOD, interval=config.DATA_INTERVAL, progress=False)
            
            if not df.empty:
                # Flatten MultiIndex columns if present (yf.download returns MultiIndex)
                # This prevents "unsupported format string passed to Series.__format__" errors
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                logger.info(f"✅ Data fetched successfully: {len(df)} candles")
                return df
                
            # If empty, wait and retry
            logger.warning(f"Attempt {i+1}/{config.MAX_RETRIES}: Empty data received, retrying...")
            time.sleep(config.RETRY_DELAY_SECONDS)
            
        except Exception as e:
            logger.error(f"Attempt {i+1}/{config.MAX_RETRIES} failed: {str(e)}")
            if i == config.MAX_RETRIES - 1: # Last attempt
                st.error(f"❌ Failed to fetch data after {config.MAX_RETRIES} attempts: {str(e)}")
                st.stop()
            time.sleep(1)
            
    st.error("❌ Failed to fetch data from Yahoo Finance (Empty Data)")
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
        return False, "❌ Data kosong. Tidak bisa melakukan prediksi."
    
    # Check 2: Required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"❌ Kolom yang diperlukan tidak ada: {', '.join(missing_cols)}"
    
    # Check 3: Sufficient data points
    if len(df) < min_rows:
        return False, f"❌ Data tidak cukup untuk prediksi. Diperlukan minimal {min_rows} candle, tersedia {len(df)}."
    
    # Check 4: No NaN values in critical columns
    if df[required_cols].isnull().any().any():
        nan_cols = df[required_cols].columns[df[required_cols].isnull().any()].tolist()
        return False, f"❌ Data mengandung nilai kosong (NaN) pada kolom: {', '.join(nan_cols)}"
    
    # Check 5: Price values are positive
    if (df['Close'] <= 0).any():
        return False, "❌ Data harga mengandung nilai negatif atau nol. Data tidak valid."
    
    # Check 6: Reasonable price range (sanity check for BTC)
    min_price = df['Close'].min()
    max_price = df['Close'].max()
    if min_price < config.MIN_PRICE_USD or max_price > config.MAX_PRICE_USD:
        logger.error(f"Validation failed: Price out of range (${min_price:,.0f} - ${max_price:,.0f})")
        return False, f"❌ Harga di luar rentang wajar (${min_price:,.0f} - ${max_price:,.0f}). Kemungkinan data corrupt."
    
    # All checks passed
    logger.info("✅ Data validation passed")
    return True, "✅ Data valid untuk prediksi."

# ==================== FEATURE ENGINEERING ====================
@st.cache_data(ttl=config.CACHE_TTL_INDICATORS, show_spinner=False)
def calculate_technical_indicators(df):
    """
    Calculate technical indicators for BOTH models:
    - Model V1 (4 features): Close, RSI, MACD, Signal
    - Model V2 (6 features): + ATR, Log Volume
    
    Cached for 5 minutes to improve performance.
    Returns: (df_features, df_model_v1, df_model_v2)
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
    
    # ATR (14) - For Model V2
    df_features['ATR_14'] = calculate_atr(df_features, period=config.ATR_LENGTH)
    logger.debug(f"ATR calculated: {df_features['ATR_14'].iloc[-1]:.2f}")
    
    # Log Volume - For Model V2
    df_features['Log_Volume'] = np.log(df_features['Volume'] + 1)
    logger.debug(f"Log Volume calculated: {df_features['Log_Volume'].iloc[-1]:.2f}")
    
    df_features = df_features.dropna()
    
    # Model V1: 4 features (Close, RSI, MACD, Signal)
    df_model_v1 = df_features[['Close', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9']].copy()
    
    # Model V2: 6 features (+ ATR, Log Volume)
    df_model_v2 = df_features[['Close', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'ATR_14', 'Log_Volume']].copy()
    
    logger.info(f"Technical indicators calculated. V1: {len(df_model_v1)} rows, V2: {len(df_model_v2)} rows")
    return df_features, df_model_v1, df_model_v2

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
🔴 <b>RSI OVERBOUGHT ALERT</b>

📊 RSI: {rsi_val:.1f} (>{config.ALERT_RSI_OVERBOUGHT})
💰 BTC Price: ${current_price:,.2f}

⚠️ Market mungkin jenuh beli. Potensi koreksi turun.

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        success, _ = send_telegram_message(bot_token, chat_id, message)
        if success:
            alerts_sent.append("RSI Overbought")
    
    # RSI Oversold Alert
    if alert_settings.get('rsi_oversold', False) and rsi_val < config.ALERT_RSI_OVERSOLD:
        message = f"""
🟢 <b>RSI OVERSOLD ALERT</b>

📊 RSI: {rsi_val:.1f} (<{config.ALERT_RSI_OVERSOLD})
💰 BTC Price: ${current_price:,.2f}

✅ Market mungkin jenuh jual. Potensi rebound naik.

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        success, _ = send_telegram_message(bot_token, chat_id, message)
        if success:
            alerts_sent.append("RSI Oversold")
    
    # MACD Crossover Alert
    if alert_settings.get('macd_crossover', False):
        hist = macd_val - signal_val
        if abs(hist) < 5:  # Close to crossover
            trend = "BULLISH 📈" if hist > 0 else "BEARISH 📉"
            message = f"""
🔔 <b>MACD SIGNAL</b>

📊 MACD: {macd_val:.2f}
📉 Signal: {signal_val:.2f}
📊 Histogram: {hist:.2f}

{trend}

💰 BTC Price: ${current_price:,.2f}

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
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


# ==================== LSTM PREDICTION ====================
def predict_next_price(df_model, model, scaler, sequence_length=config.SEQUENCE_LENGTH):
    logger.info(f"Starting LSTM prediction with sequence length: {sequence_length}")
    
    if len(df_model) < sequence_length:
        logger.error(f"Insufficient data: {len(df_model)} rows (need {sequence_length})")
        st.error(f"❌ Data insufficient! Need {sequence_length} rows.")
        return None, None, None
    
    # Take last 60 rows
    last_sequence = df_model.iloc[-sequence_length:].values
    last_sequence_scaled = scaler.transform(last_sequence)
    X_input = last_sequence_scaled.reshape(1, sequence_length, 4)
    
    logger.debug(f"Input shape: {X_input.shape}")
    prediction_scaled = model.predict(X_input, verbose=0)
    
    # Inverse Transform Logic
    dummy_array = np.zeros((1, 4))
    dummy_array[0, 0] = prediction_scaled[0, 0]
    predicted_price = scaler.inverse_transform(dummy_array)[0, 0]
    
    logger.info(f"Raw prediction: ${predicted_price:,.2f}")
    
    # Confidence Calculation (Standard deviation/Trend based)
    recent_prices = df_model['Close'].iloc[-10:].values
    price_changes = np.diff(recent_prices)
    volatility = np.std(price_changes)
    trend_consistency = np.abs(np.sum(np.sign(price_changes))) / len(price_changes)
    
    volatility_factor = min(volatility / np.mean(recent_prices) * 100, 1.0)
    confidence = config.CONFIDENCE_BASE + (trend_consistency * config.CONFIDENCE_TREND_WEIGHT) - (volatility_factor * config.CONFIDENCE_VOLATILITY_WEIGHT)
    confidence = max(config.CONFIDENCE_MIN, min(config.CONFIDENCE_MAX, confidence))
    
    logger.info(f"Confidence score: {confidence:.1f}% (volatility: {volatility:.2f}, trend: {trend_consistency:.2f})")
    
    current_price = df_model['Close'].iloc[-1]
    avg_move = np.mean(np.abs(price_changes))
    
    scenarios = {
        'best': predicted_price + (avg_move * 1.5),
        'worst': predicted_price - (avg_move * 1.5),
        'likely': (predicted_price * 0.7) + (current_price * 0.3)
    }
    
    logger.info(f"Prediction complete: ${predicted_price:,.2f} (Confidence: {confidence:.1f}%)")
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
        st.error(f"❌ Data insufficient! Need {sequence_length} rows.")
        return None, None, None
    
    # Take last 60 rows (6 features)
    last_sequence = df_model.iloc[-sequence_length:].values
    last_sequence_scaled = scaler.transform(last_sequence)
    X_input = last_sequence_scaled.reshape(1, sequence_length, 6)  # 6 features!
    
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
    
    # Confidence Calculation (same as V1)
    recent_prices = df_model['Close'].iloc[-10:].values
    price_changes = np.diff(recent_prices)
    volatility = np.std(price_changes)
    trend_consistency = np.abs(np.sum(np.sign(price_changes))) / len(price_changes)
    
    volatility_factor = min(volatility / np.mean(recent_prices) * 100, 1.0)
    confidence = config.CONFIDENCE_BASE + (trend_consistency * config.CONFIDENCE_TREND_WEIGHT) - (volatility_factor * config.CONFIDENCE_VOLATILITY_WEIGHT)
    confidence = max(config.CONFIDENCE_MIN, min(config.CONFIDENCE_MAX, confidence))
    
    logger.info(f"V2 Confidence score: {confidence:.1f}% (volatility: {volatility:.2f}, trend: {trend_consistency:.2f})")
    
    current_price = df_model['Close'].iloc[-1]
    avg_move = np.mean(np.abs(price_changes))
    
    scenarios = {
        'best': predicted_price + (avg_move * 1.5),
        'worst': predicted_price - (avg_move * 1.5),
        'likely': (predicted_price * 0.7) + (current_price * 0.3)
    }
    
    logger.info(f"V2 Prediction complete: ${predicted_price:,.2f} (Confidence: {confidence:.1f}%)")
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
        marker=dict(size=5)
    ))
    
    # Highlight 10 candle terakhir (paling berpengaruh)
    last_10 = pattern_data.iloc[-10:]
    fig.add_trace(go.Scatter(
        x=list(range(len(pattern_data)-10, len(pattern_data))),
        y=last_10['Close'],
        mode='markers',
        name='Recent Trend (Most Important)',
        marker=dict(size=10, color='#FF6B6B', symbol='circle')
    ))
    
    fig.update_layout(
        xaxis_title="Candle Index (0 = Oldest, 59 = Current)",
        yaxis_title="Price (USD)",
        height=500,  # Increased from 350 to 500
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0.05)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0', size=12),
        margin=dict(l=50, r=50, t=20, b=50)
    )
    
    # Add grid for better readability
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    return fig

# ==================== VISUALISASI CHART UTAMA ====================
def create_main_chart(df, df_features):
    # Slice last 100 candles
    df_viz = df.iloc[-100:]
    df_feat = df_features.iloc[-100:]
    
    from plotly.subplots import make_subplots
    
    # Create Subplots: Price, RSI, MACD, ATR (New for V2!)
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=('Price Action', 'RSI (14)', 'MACD (12,26,9)', 'ATR (14) - V2 Feature')
    )
    
    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df_viz.index,
        open=df_viz['Open'], high=df_viz['High'], low=df_viz['Low'], close=df_viz['Close'],
        name='Bitcoin',
        increasing_line_color='#00FF88', decreasing_line_color='#FF3B69'
    ), row=1, col=1)
    
    # 2. RSI
    fig.add_trace(go.Scatter(
        x=df_feat.index, y=df_feat['RSI_14'],
        name='RSI',
        line=dict(color='#BD00FF', width=2)
    ), row=2, col=1)
    
    # RSI Levels
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255, 59, 105, 0.5)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0, 255, 136, 0.5)", row=2, col=1)
    
    # 3. MACD with Histogram
    histogram = df_feat['MACD_12_26_9'] - df_feat['MACDs_12_26_9']
    
    # MACD Histogram (Bar Chart)
    colors = ['#00FF88' if val >= 0 else '#FF3B69' for val in histogram]
    fig.add_trace(go.Bar(
        x=df_feat.index, y=histogram,
        name='Histogram',
        marker_color=colors,
        opacity=0.5
    ), row=3, col=1)
    
    # MACD Line
    fig.add_trace(go.Scatter(
        x=df_feat.index, y=df_feat['MACD_12_26_9'],
        name='MACD',
        line=dict(color='#00D9FF', width=2)
    ), row=3, col=1)
    
    # Signal Line
    fig.add_trace(go.Scatter(
        x=df_feat.index, y=df_feat['MACDs_12_26_9'],
        name='Signal',
        line=dict(color='#FF9900', width=2)
    ), row=3, col=1)
    
    # 4. ATR (New for Model V2!)
    if 'ATR_14' in df_feat.columns:
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat['ATR_14'],
            name='ATR',
            line=dict(color='#00D9FF', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 217, 255, 0.1)'
        ), row=4, col=1)
        
        # Add average line for reference
        atr_avg = df_feat['ATR_14'].mean()
        fig.add_hline(
            y=atr_avg, 
            line_dash="dot", 
            line_color="rgba(255, 255, 255, 0.3)", 
            row=4, col=1,
            annotation_text=f"Avg: {atr_avg:.2f}",
            annotation_position="right"
        )
    
    fig.update_layout(
        height=800,  # Increased height for 4 subplots
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, color='#888'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#888'),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode='x unified'
    )
    
    return fig

# ==================== MAIN APP LAYOUT ====================
def main():
    # --- Sidebar ---
    with st.sidebar:
        # Rotating Logo in Sidebar
        logo_b64 = get_bitcoin_logo_base64()
        st.markdown(f"""
            <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                <img src="data:image/svg+xml;base64,{logo_b64}" 
                     style="width: 120px; height: 120px; animation: spin 4s linear infinite;">
            </div>
            <style>
                @keyframes spin {{ 
                    from {{ transform: rotate(0deg); }} 
                    to {{ transform: rotate(360deg); }} 
                }}
            </style>
        """, unsafe_allow_html=True)
        
        st.title("⚡ SKRIPSI DASHBOARD")
        st.info("**Algoritma:** LSTM\n**Indikator:** RSI & MACD")
        
        st.write("---")
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        model_version = st.radio(
            "Choose LSTM Model:",
            options=["V1 (4 Features)", "V2 (6 Features - Enhanced)"],
            index=0,
            help="V1: Close, RSI, MACD, Signal\nV2: + ATR, Log Volume"
        )
        
        # Dynamic Model Info based on selection
        st.subheader("📊 Model Info")
        if model_version == "V1 (4 Features)":
            st.markdown("""
            **Version:** V1 (Baseline)  
            **Architecture:** LSTM (60 timesteps)  
            **Features (4):** Close, RSI, MACD, Signal  
            **Training Period:** Historical BTC Data  
            **Prediction Horizon:** 15 Minutes  
            **Estimated Accuracy:** 55-65%  
            """)
        else:
            if model_v2 is not None:
                st.markdown("""
                **Version:** V2 (Enhanced) ✨  
                **Architecture:** LSTM (60 timesteps)  
                **Features (6):** Close, RSI, MACD, Signal, **ATR**, **Log Volume**  
                **Training Period:** Historical BTC Data  
                **Prediction Horizon:** 15 Minutes  
                **Estimated Accuracy:** 60-70% (Improved!)  
                """)
                st.success("✅ Model V2 loaded and ready!")
            else:
                st.warning("⚠️ Model V2 files not found. Using V1 instead.")
                model_version = "V1 (4 Features)"  # Fallback to V1
        
        st.write("---")
        
        # Model Performance Tracker (Minimal Backtesting)
        st.subheader("📊 Model Performance")
        
        # Initialize performance tracking in session state from persistent storage
        if 'performance_tracker' not in st.session_state:
            st.session_state['performance_tracker'] = load_tracker_data()
        
        tracker = st.session_state['performance_tracker']
        
        # Calculate accuracy if we have data (V1 OR V2)
        if len(tracker['v1_predictions']) > 0 or len(tracker['v2_predictions']) > 0:
            v1_accuracy = (tracker['v1_correct'] / len(tracker['v1_predictions'])) * 100 if len(tracker['v1_predictions']) > 0 else 0
            v2_accuracy = (tracker['v2_correct'] / len(tracker['v2_predictions'])) * 100 if len(tracker['v2_predictions']) > 0 else 0
            
            # Display performance
            perf_col1, perf_col2 = st.columns(2)
            with perf_col1:
                st.metric(
                    "V1 Accuracy",
                    f"{v1_accuracy:.1f}%",
                    help=f"Based on {len(tracker['v1_predictions'])} predictions"
                )
            with perf_col2:
                if len(tracker['v2_predictions']) > 0:
                    delta_acc = v2_accuracy - v1_accuracy
                    st.metric(
                        "V2 Accuracy",
                        f"{v2_accuracy:.1f}%",
                        f"{delta_acc:+.1f}%",
                        help=f"Based on {len(tracker['v2_predictions'])} predictions"
                    )
                else:
                    st.metric("V2 Accuracy", "N/A", help="No V2 predictions yet")
            
            st.caption(f"📈 Tracked: {len(tracker['v1_predictions'])} predictions")
            
            # Reset button
            if st.button("🔄 Reset Tracker", use_container_width=True):
                # Delete JSON file
                if os.path.exists(TRACKER_FILE):
                    os.remove(TRACKER_FILE)
                    logger.info("Tracker data file deleted")
                
                # Reset session state
                st.session_state['performance_tracker'] = {
                    'v1_predictions': [],
                    'v2_predictions': [],
                    'v1_correct': 0,
                    'v2_correct': 0,
                    'last_actual_price': None
                }
                st.success("✅ Tracker reset successfully!")
                st.rerun()
        else:
            st.info("📊 Run predictions to start tracking performance!")
        
        st.write("---")
        st.subheader("⚙️ Settings")
        
        # Force Refresh Button
        if st.button("🔄 Force Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Cache cleared! Reloading fresh data...")
            st.rerun()
        
        # Telegram Alert Settings
        st.write("---")
        st.subheader("📱 Telegram Alerts")
        
        # Bot Credentials
        with st.expander("🔐 Bot Credentials", expanded=False):
            bot_token = st.text_input(
                "Bot Token", 
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
            if st.button("🧪 Test Connection", use_container_width=True):
                if bot_token and chat_id:
                    test_msg = f"""
🎉 <b>Connection Test Successful!</b>

✅ Bot is connected to your Telegram account.
📱 You will receive alerts here.

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
                    success, msg = send_telegram_message(bot_token, chat_id, test_msg)
                    if success:
                        st.success("✅ Test message sent! Check your Telegram.")
                    else:
                        st.error(f"❌ Failed: {msg}")
                else:
                    st.warning("⚠️ Please enter both Bot Token and Chat ID")
        
        # Alert Toggles
        st.markdown("**Alert Types:**")
        alert_rsi_overbought = st.checkbox("🔴 RSI Overbought (>70)", value=False, key="alert_rsi_ob")
        alert_rsi_oversold = st.checkbox("🟢 RSI Oversold (<30)", value=False, key="alert_rsi_os")
        alert_macd = st.checkbox("🔔 MACD Signal", value=False, key="alert_macd")
        alert_prediction = st.checkbox("🎯 Prediction Results", value=False, key="alert_pred")
        
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
        refresh = st.checkbox("Auto-Refresh (Live Mode)", value=False)
        if refresh:
            st.rerun()
        
        # Author Credit (Sidebar)
        st.write("---")
        st.markdown("""
        <div style="padding: 15px; background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(189, 0, 255, 0.1)); 
                    border-radius: 10px; border: 1px solid rgba(0, 217, 255, 0.3); text-align: center;">
            <div style="font-size: 0.7rem; color: #888; margin-bottom: 5px;">DEVELOPED BY</div>
            <div style="font-size: 1rem; font-weight: bold; color: #00D9FF; margin-bottom: 3px;">Ahmad Nur Fauzan</div>
            <div style="font-size: 0.8rem; color: #BD00FF;">NIM: 2209106057</div>
            <div style="font-size: 0.7rem; color: #888; margin-top: 8px;">Skripsi - Informatika</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Header ---
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        # Static logo in header
        logo_b64 = get_bitcoin_logo_base64()
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 20px;">
                <img src="data:image/svg+xml;base64,{logo_b64}" style="width: 60px; height: 60px;">
                <div>
                    <h1 style="margin: 0; padding: 0; font-size: 2.5rem;">BTC Intraday Prediction</h1>
                    <p style="margin: 5px 0 0 0; color: #BBB; font-size: 1rem; font-style: italic;">
                        Rancang Bangun Dashboard Prediksi Harga Bitcoin Intraday Menggunakan LSTM Berbasis RSI & MACD
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_h2:
        # Live Clock using Components (Reliable Iframe)
        clock_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@700&display=swap" rel="stylesheet">
            <style>
                body {{
                    background-color: transparent;
                    margin: 0;
                    padding: 0;
                    text-align: right;
                    font-family: 'JetBrains Mono', monospace;
                }}
                .clock-container {{
                    padding-top: 20px;
                }}
                #clock {{
                    font-size: 24px;
                    color: #00D9FF;
                    font-weight: bold;
                    text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
                }}
                .label {{
                    font-size: 12px;
                    color: #888;
                    margin-top: 4px;
                    font-family: sans-serif;
                }}
            </style>
        </head>
        <body>
            <div class="clock-container">
                <div id="clock">Loading...</div>
                <div class="label">SERVER TIME (UTC)</div>
            </div>
            <script>
                function updateClock() {{
                    var now = new Date();
                    var time = now.toISOString().split('T')[1].split('.')[0];
                    document.getElementById('clock').innerHTML = time;
                }}
                setInterval(updateClock, 1000);
                updateClock();
            </script>
        </body>
        </html>
        """
        components.html(clock_html, height=100)

    # --- Data Loading ---
    with st.spinner("📡 Fetching Market Data..."):
        df_raw = get_live_bitcoin_data()
        df_full, df_model_v1, df_model_v2 = calculate_technical_indicators(df_raw)
    
    # Select active model based on sidebar choice
    if model_version == "V1 (4 Features)":
        df_model = df_model_v1
        model_active = model
        scaler_active = scaler
        predict_func = predict_next_price
        logger.info("Using Model V1 (4 features)")
    else:
        if model_v2 is not None and scaler_v2 is not None:
            df_model = df_model_v2
            model_active = model_v2
            scaler_active = scaler_v2
            predict_func = predict_next_price_v2
            logger.info("Using Model V2 (6 features)")
        else:
            # Fallback to V1 if V2 not available
            df_model = df_model_v1
            model_active = model
            scaler_active = scaler
            predict_func = predict_next_price
            logger.warning("Model V2 not available, falling back to V1")

    # --- Metrics Row ---
    current_price = df_raw['Close'].iloc[-1]
    prev_price = df_raw['Close'].iloc[-2]
    delta = current_price - prev_price
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            "Bitcoin Price", 
            f"${current_price:,.2f}", 
            f"{delta:+.2f}",
            help="Harga Bitcoin saat ini dalam USD. Delta menunjukkan perubahan dari candle sebelumnya (15 menit yang lalu)."
        )
    with m2:
        rsi_val = df_full['RSI_14'].iloc[-1]
        rsi_status = "🔴 Overbought" if rsi_val > 70 else "🟢 Oversold" if rsi_val < 30 else "⚪ Netral"
        st.metric(
            "RSI (14)", 
            f"{rsi_val:.1f}", 
            delta=None,
            help=f"Relative Strength Index (14 periode). Nilai saat ini: {rsi_status}. "
                 f"RSI > 70 = Overbought (potensi turun), RSI < 30 = Oversold (potensi naik)."
        )
    with m3:
        # MACD Histogram as delta
        macd_val = df_full['MACD_12_26_9'].iloc[-1]
        signal_val = df_full['MACDs_12_26_9'].iloc[-1]
        hist = macd_val - signal_val
        momentum = "🟢 Bullish" if hist > 0 else "🔴 Bearish"
        st.metric(
            "MACD", 
            f"{macd_val:.2f}", 
            f"{hist:.2f} (Hist)",
            help=f"Moving Average Convergence Divergence (12,26,9). Histogram: {momentum}. "
                 f"Histogram > 0 = Momentum naik, Histogram < 0 = Momentum turun."
        )
    
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
    
    
    # Timestamp Info
    last_candle_time = df_raw.index[-1]
    next_candle_time = last_candle_time + timedelta(minutes=15)
    
    st.caption(f"📅 **Data Terakhir:** {last_candle_time.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
               f"🔮 **Prediksi Untuk:** {next_candle_time.strftime('%H:%M')} UTC")

    # --- Main Chart ---
    st.plotly_chart(create_main_chart(df_raw, df_full), use_container_width=True)

    # --- Prediction Core ---
    st.markdown("### 🧬 LSTM Prediction Core")
    
    col_pred_btn, col_pred_res = st.columns([1, 3])
    
    with col_pred_btn:
        if st.button("🚀 RUN PREDICTION MODEL", use_container_width=True, type="primary"):
            # Validate data first
            is_valid, validation_msg = validate_data_for_prediction(df_raw, min_rows=60)
            
            if not is_valid:
                st.error(validation_msg)
                st.warning("💡 **Saran:** Coba klik tombol 'Force Refresh Data' di sidebar untuk mendapatkan data terbaru.")
            else:
                # Data valid, proceed with prediction
                try:
                    with st.spinner(f"⚡ Running LSTM {model_version} Inference..."):
                        pred_price, conf, scenarios = predict_func(df_model, model_active, scaler_active)
                        
                        if pred_price:
                            diff = pred_price - current_price
                            pct_diff = (diff / current_price) * 100
                            
                            st.session_state['last_pred'] = {
                                'price': pred_price, 'conf': conf, 'scenarios': scenarios,
                                'diff': diff, 'pct': pct_diff
                            }
                            st.success("✅ Prediksi berhasil!")
                            
                            # Track prediction for performance monitoring
                            tracker = st.session_state.get('performance_tracker', {
                                'v1_predictions': [],
                                'v2_predictions': [],
                                'v1_correct': 0,
                                'v2_correct': 0,
                                'last_actual_price': None
                            })
                            
                            # Store prediction with metadata
                            prediction_record = {
                                'timestamp': datetime.now(),
                                'predicted_price': pred_price,
                                'current_price': current_price,
                                'model_version': model_version,
                                'direction': 'up' if diff > 0 else 'down'
                            }
                            
                            if model_version == "V1 (4 Features)":
                                tracker['v1_predictions'].append(prediction_record)
                            else:
                                tracker['v2_predictions'].append(prediction_record)
                            
                            # Update tracker in session state
                            st.session_state['performance_tracker'] = tracker
                            
                            # Save to persistent storage
                            save_tracker_data(tracker)
                            
                            logger.info(f"Prediction tracked: {model_version} - ${pred_price:,.2f}")
                            logger.info(f"Tracker state: V1={len(tracker['v1_predictions'])}, V2={len(tracker['v2_predictions'])}")
                            
                            # Send Telegram alert for prediction (if enabled)
                            if st.session_state.get('alert_settings', {}).get('prediction', False):
                                bot_token = st.session_state.get('telegram_bot_token', '')
                                chat_id = st.session_state.get('telegram_chat_id', '')
                                
                                if bot_token and chat_id:
                                    direction = "NAIK 📈" if diff > 0 else "TURUN 📉"
                                    conf_level = "TINGGI" if conf >= 70 else "SEDANG" if conf >= 55 else "RENDAH"
                                    pred_time = (datetime.now() + timedelta(minutes=15)).strftime('%H:%M')
                                    
                                    pred_msg = f"""
🎯 <b>LSTM PREDICTION ALERT</b>

💰 Current Price: ${current_price:,.2f}
🔮 Predicted Price: ${pred_price:,.2f}

📊 Change: {direction} {abs(pct_diff):.2f}%
💵 Difference: ${abs(diff):,.2f}

🎲 Confidence: {conf_level} ({conf:.1f}%)

📈 Scenarios:
  • Best: ${scenarios['best']:,.2f}
  • Likely: ${scenarios['likely']:,.2f}
  • Worst: ${scenarios['worst']:,.2f}

🕐 Prediction Time: {pred_time} UTC

⚠️ Disclaimer: For reference only, not financial advice.
"""
                                    send_telegram_message(bot_token, chat_id, pred_msg)
                        else:
                            st.error("❌ Model gagal menghasilkan prediksi. Silakan coba lagi.")
                            
                except Exception as e:
                    st.error(f"❌ **Error saat prediksi:** {str(e)}")
                    st.warning("💡 **Troubleshooting:**\n"
                              "1. Pastikan file model (`model_bitcoin_final.keras`) ada\n"
                              "2. Pastikan file scaler (`scaler_bitcoin.pkl`) ada\n"
                              "3. Coba refresh data dengan tombol di sidebar")
                    # Optional: Log error for debugging
                    import traceback
                    with st.expander("🔍 Detail Error (untuk debugging)"):
                        st.code(traceback.format_exc())

    with col_pred_res:
        if 'last_pred' in st.session_state:
            res = st.session_state['last_pred']
            
            # Prediction Cards
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                st.markdown(f"""
                <div style="padding: 15px; background: rgba(0, 217, 255, 0.1); border-radius: 10px; border-left: 4px solid #00D9FF;">
                    <div style="color: #888; font-size: 0.8rem;">PREDICTED PRICE (+15m)</div>
                    <div style="font-size: 1.8rem; font-weight: bold; color: #FFF;">${res['price']:,.2f}</div>
                    <div style="color: {'#00FF88' if res['diff']>0 else '#FF3B69'}; font-weight: bold;">
                        {res['diff']:+.2f} ({res['pct']:+.2f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with pc2:
                 st.markdown(f"""
                <div style="padding: 15px; background: rgba(189, 0, 255, 0.1); border-radius: 10px; border-left: 4px solid #BD00FF;">
                    <div style="color: #888; font-size: 0.8rem;">MODEL CONFIDENCE</div>
                    <div style="font-size: 1.8rem; font-weight: bold; color: #FFF;">{res['conf']:.1f}%</div>
                    <div style="color: #BD00FF; font-size: 0.8rem;">Based on Sequence Stability</div>
                </div>
                """, unsafe_allow_html=True)
                 
            with pc3:
                st.markdown(f"""
                <div style="padding: 15px; background: rgba(255, 255, 255, 0.05); border-radius: 10px;">
                    <div style="color: #888; font-size: 0.8rem;">SCENARIO ANALYSIS</div>
                    <div style="font-size: 0.9rem; margin-top: 5px;">
                        <span style="color: #00FF88;">🚀 Best: ${res['scenarios']['best']:,.0f}</span><br>
                        <span style="color: #E0E0E0;">🎯 Likely: ${res['scenarios']['likely']:,.0f}</span><br>
                        <span style="color: #FF3B69;">⚠️ Worst: ${res['scenarios']['worst']:,.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            
            # --- Interpretation & Disclaimer ---
            st.markdown("---")
            
            # Interpretasi dengan confidence
            confidence_text = "TINGGI" if res['conf'] >= 70 else "SEDANG" if res['conf'] >= 55 else "RENDAH"
            direction_text = "NAIK" if res['diff'] > 0 else "TURUN"
            direction_emoji = "📈" if res['diff'] > 0 else "📉"
            
            # Waktu prediksi (+15 menit dari sekarang)
            pred_time = (datetime.now() + timedelta(minutes=15)).strftime('%H:%M')
            
            # Interpretasi AI (Full Width)
            st.info(f"""
            **💡 Interpretasi AI:**  
            Berdasarkan analisis pola LSTM, model memprediksi bahwa dalam **15 menit ke depan** (sekitar pukul **{pred_time}**),  
            harga Bitcoin berpotensi **{direction_text} {direction_emoji}** sebesar **{abs(res['pct']):.2f}%** menuju level **${res['price']:,.2f}**.
            
            **Tingkat Keyakinan (Confidence): {confidence_text} ({res['conf']:.1f}%)**
            """)
            
            # Disclaimer (Full Width)
            st.warning("""
            **⚠️ Disclaimer Penting:**
            1. **Akurasi Model:** LSTM untuk timeframe 15 menit memiliki volatilitas tinggi (akurasi estimasi 55-65%).
            2. **Data Input:** Prediksi ini murni berdasarkan pola historis **60 candle terakhir** (15 jam ke belakang).
            3. **Faktor Eksternal:** Market crypto sangat dipengaruhi news/event global yang tidak bisa dilihat oleh model ini.
            4. **Bukan Nasihat Finansial:** Gunakan data ini sebagai referensi pendukung keputusan, bukan acuan tunggal.
            """)
            
            # Pattern Visualizer (Full Width, Larger)
            st.markdown("---")
            st.subheader("🔍 60-Candle Pattern (Input Model LSTM)")
            st.caption("Grafik ini menunjukkan 60 data point terakhir yang 'dilihat' oleh model sebelum membuat prediksi. "
                      "Model LSTM menganalisis pola Close, RSI, MACD, dan Signal dari 60 candle ini untuk memprediksi harga berikutnya.")
            
            # Create larger pattern chart
            pattern_fig = create_pattern_chart(df_model)
            st.plotly_chart(pattern_fig, use_container_width=True)
            
            # CSV Export Feature
            st.markdown("---")
            st.subheader("📥 Export Hasil Prediksi")
            
            # Prepare export data with readable column names
            export_data = {
                'Date': [datetime.now().strftime('%Y-%m-%d')],
                'Time': [datetime.now().strftime('%H:%M:%S')],
                'Current_Price_USD': [round(current_price, 2)],
                'Predicted_Price_USD': [round(res['price'], 2)],
                'Price_Change_USD': [round(res['diff'], 2)],
                'Price_Change_Percent': [round(res['pct'], 2)],
                'Confidence_Percent': [round(res['conf'], 1)],
                'Best_Case_USD': [round(res['scenarios']['best'], 2)],
                'Likely_Case_USD': [round(res['scenarios']['likely'], 2)],
                'Worst_Case_USD': [round(res['scenarios']['worst'], 2)],
                'RSI_14': [round(df_full['RSI_14'].iloc[-1], 2)],
                'MACD': [round(macd_val, 4)],
                'MACD_Signal': [round(signal_val, 4)]
            }
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="💾 Download Prediction Report (CSV)",
                data=csv,
                file_name=f"btc_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.02); border-radius: 10px; margin-top: 30px;">
        <div style="font-size: 1.1rem; font-weight: bold; color: #00D9FF; margin-bottom: 10px;">
            📊 Bitcoin LSTM Prediction Dashboard
        </div>
        <div style="font-size: 0.9rem; color: #BBB; margin-bottom: 15px;">
            Rancang Bangun Dashboard Prediksi Harga Bitcoin Intraday Menggunakan LSTM Berbasis RSI & MACD
        </div>
        <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; margin-bottom: 15px;">
            <div style="font-size: 0.85rem; color: #888;">
                <span style="color: #BD00FF;">👨‍💻 Developer:</span> Ahmad Nur Fauzan
            </div>
            <div style="font-size: 0.85rem; color: #888;">
                <span style="color: #BD00FF;">🎓 NIM:</span> 2209106057
            </div>
            <div style="font-size: 0.85rem; color: #888;">
                <span style="color: #BD00FF;">🏛️ Program Studi:</span> Informatika
            </div>
        </div>
        <div style="font-size: 0.75rem; color: #666; margin-top: 10px;">
            Tech Stack: LSTM + RSI + MACD | Data Source: Yahoo Finance | Framework: Streamlit
        </div>
        <div style="font-size: 0.7rem; color: #555; margin-top: 8px;">
            © 2025 Skripsi Project - All Rights Reserved
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
