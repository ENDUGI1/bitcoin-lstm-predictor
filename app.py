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

# ==================== CUSTOM CSS (CYBERPUNK STYLE) ====================
def inject_custom_css():
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;500;700&family=JetBrains+Mono:wght@400;700&display=swap');

        /* --- GLOBAL VARIABLES --- */
        :root {
            --primary-color: #00D9FF;
            --secondary-color: #BD00FF;
            --success-color: #00FF88;
            --danger-color: #FF3B69;
            --bg-color: #0E1117;
            --card-bg: rgba(18, 22, 31, 0.7);
            --card-border: 1px solid rgba(255, 255, 255, 0.08);
            --glass-effect: blur(12px);
        }

        /* --- TYPOGRAPHY --- */
        html, body, [class*="css"] {
            font-family: 'Rajdhani', sans-serif !important;
            letter-spacing: 0.5px;
        }

        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif !important;
            text-transform: uppercase;
            letter-spacing: 2px;
            background: linear-gradient(135deg, #FFF 0%, var(--primary-color) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(0, 217, 255, 0.2);
        }

        /* --- MAIN CONTAINER --- */
        .stApp {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(189, 0, 255, 0.05) 0%, transparent 20%),
                radial-gradient(circle at 90% 80%, rgba(0, 217, 255, 0.05) 0%, transparent 20%);
        }

        /* --- METRICS CARDS ('stMetric') --- */
        div[data-testid="stMetric"] {
            background: var(--card-bg);
            border: var(--card-border);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: var(--glass-effect);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        div[data-testid="stMetric"]::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 2px;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            opacity: 0.5;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 217, 255, 0.15);
            border-color: rgba(0, 217, 255, 0.3);
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
            color: #8899A6 !important;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 1px;
        }

        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.2rem !important;
            font-weight: 700;
            color: #FFF !important;
            text-shadow: 0 0 10px rgba(255,255,255,0.1);
        }

        div[data-testid="stMetricDelta"] {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(0,0,0,0.3);
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.8rem !important;
        }

        /* --- EXPANDERS & CONTAINERS --- */
        div[data-testid="stExpander"] {
            background: transparent;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
        }
        
        div[data-testid="stExpander"]:hover {
            border-color: var(--primary-color);
        }

        /* --- BUTTONS --- */
        div.stButton > button {
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(189, 0, 255, 0.1) 100%);
            border: 1px solid rgba(0, 217, 255, 0.5);
            color: #FFF;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 2px;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }

        div.stButton > button:hover {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            border-color: transparent;
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.4);
            transform: scale(1.02);
        }

        div.stButton > button:active {
            transform: scale(0.98);
        }

        /* --- SIDEBAR --- */
        section[data-testid="stSidebar"] {
            background-color: #050505;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        section[data-testid="stSidebar"] h1 {
            background: none;
            -webkit-text-fill-color: #FFF;
            font-size: 1.2rem;
            letter-spacing: 1px;
            text-shadow: none;
        }

        /* --- ALERTS (INFO, SUCCESS, WARNING) --- */
        div[data-testid="stAlert"] {
            border-radius: 12px;
            background: rgba(18, 22, 31, 0.8);
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }

        /* --- PLOTLY CHARTS --- */
        .js-plotly-plot .plotly .main-svg {
            background: transparent !important;
        }
        
        /* --- GAMING/FUTURISTIC SCROLLBAR --- */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #0E1117; 
        }
        ::-webkit-scrollbar-thumb {
            background: #333; 
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary-color); 
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
    Create a Plotly gauge chart for confidence visualization.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': '#E0E0E0'}},
        number={'suffix': "%", 'font': {'size': 32, 'color': '#FFF'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#888"},
            'bar': {'color': "#00D9FF"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 55], 'color': 'rgba(255, 59, 105, 0.3)'},  # Red - Low
                {'range': [55, 70], 'color': 'rgba(255, 153, 0, 0.3)'},  # Orange - Medium
                {'range': [70, 100], 'color': 'rgba(0, 255, 136, 0.3)'}  # Green - High
            ],
            'threshold': {
                'line': {'color': "#BD00FF", 'width': 4},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#E0E0E0"},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
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
            if pred_time and (current_time - pred_time).total_seconds() >= 900:  # 15 min = 900 sec
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
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Directional Accuracy Over Time', 'MAE Over Time', 'RMSE Over Time'),
        vertical_spacing=0.1,
        row_heights=[0.33, 0.33, 0.33]
    )
    
    if len(preds) >= 2:
        counts, dir_vals, mae_vals, rmse_vals = calc_cumulative_metrics(preds)
        
        # Directional accuracy
        fig.add_trace(go.Scatter(
            x=counts, y=dir_vals,
            name='Directional Accuracy',
            line=dict(color='#00D9FF', width=2),
            mode='lines+markers'
        ), row=1, col=1)
        
        # MAE
        fig.add_trace(go.Scatter(
            x=counts, y=mae_vals,
            name='MAE',
            line=dict(color='#00D9FF', width=2),
            mode='lines+markers',
            showlegend=False
        ), row=2, col=1)
        
        # RMSE
        fig.add_trace(go.Scatter(
            x=counts, y=rmse_vals,
            name='RMSE',
            line=dict(color='#00D9FF', width=2),
            mode='lines+markers',
            showlegend=False
        ), row=3, col=1)
    
    # Update layout
    fig.update_xaxes(title_text="Number of Predictions", row=3, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="MAE ($)", row=2, col=1)
    fig.update_yaxes(title_text="RMSE ($)", row=3, col=1)
    
    fig.update_layout(
        height=800,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center")
    )
    
    return fig


# ==================== BACKTESTING SYSTEM ====================
def run_backtest(start_date, end_date, model_v1, scaler_v1):
    """
    Run backtesting on historical data.
    Returns comprehensive metrics and prediction history.
    """
    try:
        # Fetch historical data
        ticker = yf.Ticker("BTC-USD")
        df_hist = ticker.history(start=start_date, end=end_date, interval="15m")
        
        if df_hist.empty or len(df_hist) < 100:
            return None, "Insufficient historical data for selected date range"
        
        # Calculate technical indicators
        df_full, df_model = calculate_technical_indicators(df_hist)
        
        # Initialize results
        results = {
            'predictions': [], 'directional_correct': 0, 'total': 0
        }
        
        # Run predictions on each timestamp (skip last 1 to have actual price)
        total_points = len(df_model) - 1
        
        for i in range(60, total_points):  # Start from 60 to have enough history
            try:
                # Get data up to current point
                df_subset = df_model.iloc[:i+1]
                
                # Prediction
                pred, conf, _ = predict_next_price(df_subset, model_v1, scaler_v1)
                
                # Get actual price (next timestamp)
                actual_price = df_hist['Close'].iloc[i+1]
                current_price = df_hist['Close'].iloc[i]
                
                # Calculate errors
                if pred:
                    error = abs(pred - actual_price)
                    direction = 'up' if pred > current_price else 'down'
                    actual_direction = 'up' if actual_price > current_price else 'down'
                    
                    results['predictions'].append({
                        'predicted': pred,
                        'actual': actual_price,
                        'error': error,
                        'direction_correct': direction == actual_direction
                    })
                    
                    if direction == actual_direction:
                        results['directional_correct'] += 1
                    results['total'] += 1
                    
            except Exception as e:
                logger.warning(f"Backtest prediction failed at index {i}: {str(e)}")
                continue
        
        # Calculate final metrics
        metrics = {}
        
        preds = results['predictions']
        if len(preds) > 0:
            errors = [p['error'] for p in preds]
            
            metrics['v1'] = {
                'total_predictions': len(preds),
                'directional_accuracy': (results['directional_correct'] / results['total']) * 100,
                'mae': np.mean(errors),
                'rmse': np.sqrt(np.mean([e**2 for e in errors])),
                'min_error': np.min(errors),
                'max_error': np.max(errors),
                'median_error': np.median(errors)
            }
        else:
            metrics['v1'] = None
        
        return metrics, None
        
        logger.info("🎉 Model V2 files downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading Model V2 files: {str(e)}")
        logger.warning("Model V2 will not be available. Using V1 only.")
        return False

# ==================== LOAD MODEL & SCALER ====================
@st.cache_resource
def load_model_and_scaler():
    """Load pretrained LSTM model and scaler (4 features)"""
    try:
        model = tf.keras.models.load_model(config.MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        logger.info("✅ Model (4 features) loaded successfully")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading Model: {str(e)}")
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
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching Bitcoin data from Binance (attempt {attempt + 1}/{max_retries})")
            
            url = f"{config.BINANCE_BASE_URL}/api/v3/klines"
            params = {
                "symbol": config.BINANCE_SYMBOL,
                "interval": config.BINANCE_INTERVAL,
                "limit": config.BINANCE_LIMIT
            }
            
            response = requests.get(url, params=params, timeout=15)
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
            
            logger.info(f"✅ Binance data fetched successfully: {len(df)} candles")
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
            logger.info("📊 Using Binance data (real-time)")
            # Track data source for UI indicator
            st.session_state['data_source_used'] = 'binance'
            return df
        else:
            logger.warning("⚠️ Binance failed, falling back to yfinance...")
    
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
                
                logger.info(f"✅ yfinance data fetched successfully: {len(df)} candles")
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
    atr_factor = min(atr_normalized * 100, 1.0)  # Cap at 1.0
    
    # Calculate base factors
    volatility_factor = min(volatility / np.mean(recent_prices) * 100, 1.0)
    
    # V2 Formula: Base + Trend Bonus - Volatility Penalty - ATR Penalty
    confidence = (config.CONFIDENCE_BASE + 
                  (trend_consistency * config.CONFIDENCE_TREND_WEIGHT) - 
                  (volatility_factor * config.CONFIDENCE_VOLATILITY_WEIGHT) -
                  (atr_factor * config.CONFIDENCE_ATR_WEIGHT))  # NEW: ATR penalty
    
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
    
    # Create Subplots: Price, RSI, MACD
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=('Price Action', 'RSI (14)', 'MACD (12,26,9)')
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
    
    fig.update_layout(
        height=700,
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
        
        st.title("⚡DASHBOARD")
        st.info("**Algoritma:** LSTM\n**Indikator:** RSI & MACD")
        
        st.write("---")
        
        # Model Info
        st.subheader("📊 Model Info")
        st.markdown("""
        **Architecture:** LSTM (60 timesteps)  
        **Features (4):** Close, RSI, MACD, Signal  
        **Prediction Horizon:** 15 Minutes  
        """)

        st.write("---")

        # Model Performance Tracker
        st.subheader("📊 Model Performance")
        
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
                st.metric("Directional Accuracy", f"{metrics['directional_accuracy']:.1f}%")
                st.caption(f"MAE: ${metrics['mae']:.2f}")
                st.caption(f"RMSE: ${metrics['rmse']:.2f}")
                st.caption(f"📊 {metrics['total_predictions']} verified")
            else:
                st.metric("Accuracy", f"{accuracy:.1f}%")
                st.caption(f"📈 {len(tracker['predictions'])} predictions")
                st.caption("⏳ Awaiting actual prices")
            
            # Reset button
            if st.button("Reset Tracker", use_container_width=True):
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
                st.success("✅ Tracker reset successfully!")
                st.rerun()
            
            # Manual Update Actual Prices Button
            st.write("---")
            if st.button("Update Actual Prices", use_container_width=True, help="Manually update all predictions with current price for testing"):
                st.session_state['trigger_manual_update'] = True
                st.rerun()

        else:
            st.info("📊 Run predictions to start tracking performance!")
        
        # Prediction History & Error Analysis
        if len(tracker.get('predictions', [])) > 0:
            st.write("---")
            with st.expander("📋 Prediction History & Errors", expanded=False):
                st.caption("View individual prediction errors and identify outliers")
                
                pred_data = []
                for i, pred in enumerate(tracker['predictions'][-10:], 1):  # Last 10
                    if pred.get('actual_price') is not None:
                        error = abs(pred['predicted_price'] - pred['actual_price'])
                        direction_correct = "✅" if pred.get('direction') == ('up' if pred['actual_price'] > pred['current_price'] else 'down') else "❌"
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
                            st.warning(f"⚠️ Outliers detected: #{', #'.join(map(str, outliers))}")
                else:
                    st.caption("⏳ No verified predictions yet")

        st.write("---")
        st.subheader("⚙️ Settings")
        
        # Backtesting Section (NEW!)
        with st.expander("🧪 Backtesting Dashboard", expanded=False):
            st.caption("Test models on historical data")
            
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
            if st.button("🚀 Run Backtest", use_container_width=True):
                if start_date >= end_date:
                    st.error("❌ Start date must be before end date")
                else:
                    with st.spinner("⚡ Running backtest... This may take a few minutes..."):
                        metrics, error = run_backtest(
                            start_date, end_date,
                            model, scaler
                        )
                        
                        if error:
                            st.error(f"❌ Backtest failed: {error}")
                        elif metrics:
                            st.session_state['backtest_results'] = metrics
                            st.success("✅ Backtest complete! Scroll down to see results.")
                            st.rerun()
                        else:
                            st.error("❌ No results generated")
        

        # Force Refresh Button
        if st.button("Force Refresh Data", use_container_width=True):
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
                <img src="data:image/svg+xml;base64,{logo_b64}" style="width: 70px; height: 70px; filter: drop-shadow(0 0 10px rgba(0, 217, 255, 0.5));">
                <div>
                    <h1 style="margin: 0; padding: 0; font-size: 3rem; background: linear-gradient(135deg, #FFF 0%, var(--primary-color) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">BTC Intraday Prediction</h1>
                    <p style="margin: 5px 0 0 0; color: #8899A6; font-size: 1.1rem; letter-spacing: 1px; font-family: 'Rajdhani', sans-serif;">
                        Advanced LSTM Neural Network • RSI & MACD Strategy
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
        df_full, df_model = calculate_technical_indicators(df_raw)
    
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
            current_price = df_raw['Close'].iloc[-1]
            updated = manual_update_all_actual_prices(tracker, current_price)
            
            if updated > 0:
                save_tracker_data(tracker)
                st.session_state['performance_tracker'] = tracker
                st.success(f"✅ Updated {updated} predictions with actual prices!")
                st.info("💡 Refresh to see updated metrics")
            else:
                st.info("ℹ️ All predictions already have actual prices")
        
        # Clear the flag
        st.session_state['trigger_manual_update'] = False

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
    
    # Data Source Indicator
    data_source_used = st.session_state.get('data_source_used', 'unknown')
    if data_source_used == 'binance':
        st.markdown("""
        <div style="display: inline-block; padding: 4px 12px; border-radius: 20px; 
                    background: linear-gradient(135deg, rgba(0, 255, 136, 0.15), rgba(0, 217, 255, 0.15)); 
                    border: 1px solid rgba(0, 255, 136, 0.4); margin-bottom: 10px;">
            <span style="color: #00FF88; font-weight: bold; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
                🔗 Data Source: Binance API (Real-time)
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: inline-block; padding: 4px 12px; border-radius: 20px; 
                    background: linear-gradient(135deg, rgba(255, 153, 0, 0.15), rgba(255, 200, 0, 0.15)); 
                    border: 1px solid rgba(255, 153, 0, 0.4); margin-bottom: 10px;">
            <span style="color: #FF9900; font-weight: bold; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
                🔗 Data Source: Yahoo Finance (Delayed ~15min)
            </span>
        </div>
        """, unsafe_allow_html=True)


    # --- Main Chart ---
    st.plotly_chart(create_main_chart(df_raw, df_full), use_container_width=True)

    # --- TradingView Interactive Chart ---
    with st.expander("📊 **TradingView Interactive Chart** (Gambar Trendline, Zoom, dll)", expanded=False):
        st.caption("💡 Chart interaktif dari TradingView. Kamu bisa gambar trendline, fibonacci, dan tools lainnya!")
        render_tradingview_widget()
        
        # Link to full TradingView for drawing tools
        st.markdown("""
        <a href="https://www.tradingview.com/chart/?symbol=BINANCE:BTCUSDT" target="_blank" 
           style="display: inline-block; padding: 8px 16px; border-radius: 8px; 
                  background: linear-gradient(135deg, #131722, #1e222d); 
                  border: 1px solid rgba(0, 217, 255, 0.3); color: #00D9FF; 
                  text-decoration: none; font-family: 'Orbitron', sans-serif; 
                  font-size: 0.85rem; margin-top: 10px;">
            🎨 Buka TradingView Full (Gambar Trendline, Fibonacci, dll)
        </a>
        """, unsafe_allow_html=True)
        st.caption("💡 Klik tombol di atas untuk akses drawing tools lengkap di TradingView.com")


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
                    with st.spinner("⚡ Running LSTM Inference..."):
                        pred_price, conf, scenarios = predict_next_price(df_model, model, scaler)
                        
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
                                'predictions': [],
                                'correct': 0,
                                'last_actual_price': None
                            })
                            
                            # Store prediction with metadata
                            prediction_record = {
                                'timestamp': datetime.now(),
                                'predicted_price': pred_price,
                                'current_price': current_price,
                                'direction': 'up' if diff > 0 else 'down'
                            }
                            
                            tracker['predictions'].append(prediction_record)
                            
                            # Update tracker in session state
                            st.session_state['performance_tracker'] = tracker
                            
                            # Save to persistent storage
                            save_tracker_data(tracker)
                            
                            logger.info(f"Prediction tracked: ${pred_price:,.2f}")
                            logger.info(f"Tracker state: {len(tracker['predictions'])} predictions")
                            
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
                              "1. Pastikan file model (`model_bitcoin_v1_4features.keras`) ada\n"
                              "2. Pastikan file scaler (`scaler_bitcoin_v1.pkl`) ada\n"
                              "3. Coba refresh data dengan tombol di sidebar")
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
                <div style="padding: 20px; background: rgba(0, 217, 255, 0.05); border-radius: 16px; border: 1px solid rgba(0, 217, 255, 0.2); box-shadow: 0 0 20px rgba(0, 217, 255, 0.1);">
                    <div style="color: #8899A6; font-size: 0.85rem; font-family: 'Orbitron', sans-serif; letter-spacing: 1px;">PREDICTED PRICE (+15m)</div>
                    <div style="font-size: clamp(1.2rem, 2vw, 2.2rem); font-weight: bold; color: #FFF; font-family: 'JetBrains Mono', monospace; margin: 10px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${res['price']:,.2f}</div>
                    <div style="display: inline-block; padding: 4px 12px; border-radius: 8px; background: rgba({'0, 255, 136' if res['diff']>0 else '255, 59, 105'}, 0.15); color: {'var(--success-color)' if res['diff']>0 else 'var(--danger-color)'}; font-weight: bold; font-family: 'JetBrains Mono', monospace;">
                        {res['diff']:+.2f} ({res['pct']:+.2f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with pc2:
                conf_subtitle = "SEQUENCE STABILITY"
                st.markdown(f"""
                <div style="padding: 20px; background: rgba(189, 0, 255, 0.05); border-radius: 16px; border: 1px solid rgba(189, 0, 255, 0.2); box-shadow: 0 0 20px rgba(189, 0, 255, 0.1);">
                    <div style="color: #8899A6; font-size: 0.85rem; font-family: 'Orbitron', sans-serif; letter-spacing: 1px;">MODEL CONFIDENCE</div>
                    <div style="font-size: 2.2rem; font-weight: bold; color: #FFF; font-family: 'JetBrains Mono', monospace; margin: 10px 0;">{res['conf']:.1f}%</div>
                    <div style="color: var(--secondary-color); font-size: 0.75rem; letter-spacing: 0.5px; text-transform: uppercase;">{conf_subtitle}</div>
                </div>
                """, unsafe_allow_html=True)
                 
            with pc3:
                st.markdown(f"""
                <div style="padding: 20px; background: rgba(255, 255, 255, 0.02); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.08);">
                    <div style="color: #8899A6; font-size: 0.85rem; font-family: 'Orbitron', sans-serif; letter-spacing: 1px;">SCENARIO ANALYSIS</div>
                    <div style="font-size: 0.95rem; margin-top: 12px; font-family: 'Rajdhani', sans-serif; font-weight: 500;">
                        <div style="margin-bottom: 6px;"><span style="color: var(--success-color);">🚀 BULL:</span> <span style="font-family: 'JetBrains Mono';">${res['scenarios']['best']:,.0f}</span></div>
                        <div style="margin-bottom: 6px;"><span style="color: #8899A6;">🎯 BASE:</span> <span style="font-family: 'JetBrains Mono'; color: #FFF;">${res['scenarios']['likely']:,.0f}</span></div>
                        <div><span style="color: var(--danger-color);">⚠️ BEAR:</span> <span style="font-family: 'JetBrains Mono';">${res['scenarios']['worst']:,.0f}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence Gauge Visualization (NEW!)
            st.markdown("---")
            st.markdown("#### 📊 Confidence Visualization")
            
            gauge_col1, gauge_col2 = st.columns([1, 2])
            
            with gauge_col1:
                # Confidence gauge chart
                gauge_fig = create_confidence_gauge(res['conf'], "Model Confidence")
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with gauge_col2:
                # Confidence breakdown
                st.markdown("**Confidence Breakdown:**")
                st.markdown(f"""
                - **Base Score:** 50%
                - **Trend Consistency:** Contributes up to +30%
                - **Volatility Penalty:** Up to -20%
                """)
                
                st.caption("💡 Uses basic volatility calculation")
                
                # Confidence level indicator
                if res['conf'] >= 70:
                    st.success("🟢 **HIGH CONFIDENCE** - Strong signal")
                elif res['conf'] >= 55:
                    st.warning("🟡 **MEDIUM CONFIDENCE** - Moderate signal")
                else:
                    st.error("🔴 **LOW CONFIDENCE** - Weak signal, use caution")
            

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
    
    # Accuracy Trend Chart
    st.markdown("---")
    st.markdown("### 📈 Accuracy Trend Analysis")
    st.caption("Model accuracy evolving over time")
    
    if 'performance_tracker' in st.session_state:
        tracker = st.session_state['performance_tracker']
        trend_chart = create_accuracy_trend_chart(tracker)
        
        if trend_chart:
            st.plotly_chart(trend_chart, use_container_width=True)
            
            # Insights
            verified = len([p for p in tracker.get('predictions', []) if p.get('actual_price') is not None])
            
            st.info(f"""
            **💡 Trend Insights:**
            - Chart shows how accuracy evolves as more predictions are verified
            - {verified} verified predictions
            - Look for upward trends in directional accuracy and downward trends in MAE/RMSE
            """)
        else:
            st.info("📊 Need at least 2 verified predictions to show trends. Keep making predictions!")
    
    # Backtest Results Display
    if 'backtest_results' in st.session_state:
        bt_results = st.session_state['backtest_results']
        
        st.markdown("---")
        st.markdown("### 🧪 Backtesting Results")
        st.caption("Historical model performance on past data")
        
        if bt_results.get('v1'):
            v1 = bt_results['v1']
            st.markdown(f"""
            <div style="padding: 20px; background: rgba(0, 217, 255, 0.05); border-radius: 16px; border: 1px solid rgba(0, 217, 255, 0.2); box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <div style="color: #8899A6; font-size: 0.8rem; font-family: 'Orbitron', sans-serif; letter-spacing: 1px;">BACKTEST RESULTS</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: var(--primary-color); font-family: 'JetBrains Mono', monospace; margin: 8px 0;">Directional: {v1['directional_accuracy']:.1f}%</div>
                <div style="color: #8899A6; font-size: 0.75rem; font-family: 'Rajdhani', sans-serif;">
                    MAE: <span style="color: #FFF;">${v1['mae']:.2f}</span><br>
                    RMSE: <span style="color: #FFF;">${v1['rmse']:.2f}</span><br>
                    Predictions: {v1['total_predictions']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Insights
        st.info("""
        **💡 Backtest Insights:**
        - Results based on historical data (past performance)
        - Large sample size provides statistical significance
        - Use these metrics to validate model performance
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.02); border-radius: 10px; margin-top: 30px;">
        <div style="font-size: 1.1rem; font-weight: bold; color: #00D9FF; margin-bottom: 10px;">
            📊 Bitcoin LSTM Prediction Dashboard
        </div>
        <div style="font-size: 0.9rem; color: #BBB; margin-bottom: 15px;">
            Implementasi Algoritma LSTM dengan Indikator RSI dan MACD untuk Prediksi Harga Bitcoin Intraday Berbasis Web
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
            Tech Stack: LSTM + RSI + MACD | Data Source: Binance API + Yahoo Finance (Fallback) | Framework: Streamlit
        </div>
        <div style="font-size: 0.7rem; color: #555; margin-top: 8px;">
            © 2025 Skripsi Project - All Rights Reserved
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
