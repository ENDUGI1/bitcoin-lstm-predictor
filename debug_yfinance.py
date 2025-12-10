import yfinance as yf
import pandas as pd

print("Attempting to fetch BTC-USD data...")
try:
    # Try fetching with yf.download wrapper
    df = yf.download("BTC-USD", period="5d", interval="15m", progress=False)
    
    if df.empty:
        print("❌ Error: Returned DataFrame is empty (yf.download).")
    else:
        print("✅ Success!")
        print(f"Rows fetched: {len(df)}")
        print("Last 5 rows:")
        print(df.tail())
        
except Exception as e:
    print(f"❌ Exception occurred: {e}")
