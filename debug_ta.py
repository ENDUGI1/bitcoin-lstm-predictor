import pandas as pd
import pandas_ta as ta
import yfinance as yf

# Mock data or fetch real data
try:
    df = yf.Ticker("BTC-USD").history(period="1d", interval="15m")
    if df.empty:
        print("Empty DataFrame")
    else:
        bbands = ta.bbands(df['Close'], length=20, std=2.0)
        print("Columns returned by ta.bbands:")
        print(bbands.columns.tolist())
except Exception as e:
    print(f"Error: {e}")
