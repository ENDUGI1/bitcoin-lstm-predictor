# ==================== CONFIGURATION FILE ====================
# Bitcoin LSTM Prediction Dashboard
# Author: Ahmad Nur Fauzan (2209106057)
# ============================================================

# ==================== MODEL PARAMETERS ====================
# LSTM Model Configuration
SEQUENCE_LENGTH = 60  # Number of candles used for prediction
PREDICTION_HORIZON_MINUTES = 15  # Prediction timeframe in minutes

# Model Files
MODEL_PATH = "model_bitcoin_final.keras"
SCALER_PATH = "scaler_bitcoin.pkl"

# ==================== TECHNICAL INDICATORS ====================
# RSI Configuration
RSI_LENGTH = 14  # Standard RSI period

# MACD Configuration
MACD_FAST = 12   # Fast EMA period
MACD_SLOW = 26   # Slow EMA period
MACD_SIGNAL = 9  # Signal line period

# ==================== DATA FETCHING ====================
# Yahoo Finance Configuration
TICKER_SYMBOL = "BTC-USD"
DATA_PERIOD = "5d"  # Historical data period
DATA_INTERVAL = "15m"  # Candle interval
MAX_RETRIES = 3  # Number of retry attempts for data fetching
RETRY_DELAY_SECONDS = 1  # Delay between retries

# ==================== CACHING ====================
# Cache TTL (Time To Live) in seconds
CACHE_TTL_DATA = 60  # 1 minute for live data
CACHE_TTL_INDICATORS = 300  # 5 minutes for calculated indicators
CACHE_TTL_MODEL = None  # No expiry for model (loaded once)

# ==================== VALIDATION ====================
# Data Validation Parameters
MIN_DATA_ROWS = 60  # Minimum rows required for prediction
MIN_PRICE_USD = 100  # Minimum reasonable BTC price
MAX_PRICE_USD = 1000000  # Maximum reasonable BTC price

# ==================== VISUALIZATION ====================
# Chart Configuration
MAIN_CHART_HEIGHT = 700  # Main chart height in pixels
PATTERN_CHART_HEIGHT = 500  # Pattern chart height in pixels
CHART_CANDLES_DISPLAY = 100  # Number of candles to display in main chart

# Color Scheme (Cyberpunk Theme)
COLOR_PRIMARY = "#00D9FF"  # Cyan
COLOR_SECONDARY = "#BD00FF"  # Purple
COLOR_SUCCESS = "#00FF88"  # Green
COLOR_DANGER = "#FF3B69"  # Red
COLOR_WARNING = "#FF9900"  # Orange

# ==================== CONFIDENCE SCORING ====================
# Confidence Calculation Parameters
CONFIDENCE_BASE = 50  # Base confidence score
CONFIDENCE_TREND_WEIGHT = 30  # Weight for trend consistency
CONFIDENCE_VOLATILITY_WEIGHT = 20  # Weight for volatility penalty
CONFIDENCE_MIN = 40  # Minimum confidence score
CONFIDENCE_MAX = 90  # Maximum confidence score

# Confidence Thresholds
CONFIDENCE_HIGH_THRESHOLD = 70  # High confidence threshold
CONFIDENCE_MEDIUM_THRESHOLD = 55  # Medium confidence threshold

# ==================== LOGGING ====================
# Logging Configuration
LOG_LEVEL = "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ==================== TELEGRAM ALERTS ====================
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = ""  # Set via sidebar (keep empty for security)
TELEGRAM_CHAT_ID = ""  # Set via sidebar (keep empty for security)

# Alert Thresholds
ALERT_RSI_OVERBOUGHT = 70  # RSI threshold for overbought alert
ALERT_RSI_OVERSOLD = 30  # RSI threshold for oversold alert
ALERT_MACD_CROSSOVER_ENABLED = True  # Enable MACD crossover detection
ALERT_PREDICTION_ENABLED = True  # Send alert after prediction

# Alert Cooldown (prevent spam)
ALERT_COOLDOWN_SECONDS = 300  # 5 minutes cooldown between same alert type

# ==================== APP METADATA ====================
# Application Information
APP_TITLE = "BTC LSTM Predictor (RSI & MACD)"
APP_SUBTITLE = "Rancang Bangun Dashboard Prediksi Harga Bitcoin Intraday Menggunakan LSTM Berbasis RSI & MACD"
AUTHOR_NAME = "Ahmad Nur Fauzan"
AUTHOR_NIM = "2209106057"
AUTHOR_PROGRAM = "Informatika"
COPYRIGHT_YEAR = "2025"
