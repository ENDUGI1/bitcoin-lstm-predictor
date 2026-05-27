# Bitcoin Intraday Forecast

A stacked LSTM neural network that projects the next 15-minute close price of
Bitcoin from RSI and MACD features. Built as an undergraduate thesis project
and shipped as a Streamlit dashboard with live data, tracked forecasts, and a
historical backtesting engine.

**[Live demo](https://btc-predict-lstm-057.streamlit.app/)**
&nbsp;·&nbsp;
**[Training notebook (Colab)](https://colab.research.google.com/drive/1-M-irrfB0srhtwqKLY9F_n3gtY1FKn80?usp=sharing)**
&nbsp;·&nbsp;
**[Dataset (Kaggle)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)**

<!--
  Drop a hero screenshot of the dashboard at `docs/hero.png` to render it here.
  Recommended size: 1600×900, dark background, taken from Edge or Firefox after
  running a prediction so the LSTM forecast card is populated.
-->
<!-- ![Dashboard](./docs/hero.png) -->

---

## Overview

Most ML crypto-prediction projects ship as Jupyter notebooks. This one is a
running product: a single-page dashboard that pulls real-time BTC candles every
15 minutes, feeds them through a trained LSTM, and renders the forecast next to
the model's own running accuracy.

The interface is built around a restrained, editorial design system called
**Quiet Quant** (see `styles.py`). The goal is to look like a serious trading
tool — not a generic AI demo.

## The model

| | |
|---|---|
| **Architecture** | Stacked LSTM, 60-timestep input sequences |
| **Input features** | `Close`, `RSI (14)`, `MACD (12·26·9)`, `MACD signal` |
| **Output** | Next 15-minute close price (single scalar) |
| **Training data** | BTC/USD historical 1-minute candles (Kaggle), resampled to 15 minutes — coverage through ~Dec 2025 |
| **Inference time** | ~1 second on commodity CPU |

RSI is computed with **Wilder's smoothing** so it matches `pandas_ta` exactly —
the indicators produced live are bit-identical to those the model was trained
on. This avoids a common source of train/serve skew in TA-based models.

## Live pipeline

```
Binance API (data-api.binance.vision)  ─┐
                                         ├─►  DataFrame  ─►  RSI + MACD
Yahoo Finance  (fallback)                ┘                       │
                                                                 ▼
                                                MinMaxScaler.transform
                                                                 │
                                                                 ▼
                                                   LSTM.predict (60×4 → 1)
                                                                 │
                                                                 ▼
                                                Plotly + custom HTML render
```

Each forecast is persisted to `tracker_data.json` and matched against the
actual close after 15 minutes, which produces a live MAE, RMSE, and
directional accuracy that the sidebar surfaces in real time.

## Features

- **Real-time forecast** — Binance primary, Yahoo Finance fallback if the
  primary times out (6 s budget).
- **Performance tracker** — every forecast is verified after 15 minutes and
  contributes to a running MAE / RMSE / directional accuracy chart.
- **Backtest engine** — replay the model over an arbitrary historical date
  range with vectorised batch inference (40-100× faster than the naive sliding
  window).
- **TradingView drill-down** — opt-in embed for manual trendline / Fibonacci
  annotation (lazy-mounted so it doesn't slow first paint).
- **Telegram alerts** — optional bot integration for RSI threshold, MACD
  crossover, and prediction notifications.
- **Quiet Quant design system** — single accent colour, two typefaces, strict
  4/8 pt grid, WCAG AA contrast, full `prefers-reduced-motion` support.

## Tech stack

| Layer | Tools |
|---|---|
| Inference | TensorFlow 2.15 (Keras), scikit-learn (`MinMaxScaler`) |
| Numerical | NumPy, pandas |
| UI | Streamlit ≥ 1.29 · custom CSS module (`styles.py`) |
| Charts | Plotly ≥ 5.18 (custom Quiet Quant theme) |
| Data | Binance public REST API · Yahoo Finance (`yfinance`) fallback |
| Persistence | JSON file (`tracker_data.json`) |
| Optional | Telegram Bot API for alerts |

## Quick start

```bash
git clone https://github.com/ENDUGI1/bitcoin-lstm-predictor.git
cd bitcoin-lstm-predictor

# create a virtualenv (recommended)
python -m venv .venv

# activate it
#   Windows (PowerShell):
.venv\Scripts\Activate.ps1
#   macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. If the page hangs on first load and you've had
Streamlit running on this port before, the cause is almost always a stale
browser service-worker or extension blocking the Plotly bundle. Two quick
workarounds:

- Use a different port: `streamlit run app.py --server.port 8888`
- Or open the URL in a fresh browser (Edge, Firefox, or Chrome incognito)

## Project structure

```
.
├── app.py                            # Streamlit UI, layout, callbacks, glue
├── styles.py                         # Quiet Quant: tokens, CSS, Plotly theme, SVG icons
├── config.py                         # Model + app parameters
├── model_bitcoin_v1_4features.keras  # Trained Keras model (4 features, 60-step input)
├── scaler_bitcoin_v1.pkl             # Fitted MinMaxScaler
├── tracker_data.json                 # Persistent forecast history (auto-managed)
├── requirements.txt                  # Python dependencies
└── .streamlit/
    └── config.toml                   # Streamlit theme + server config
```

## Design system — Quiet Quant

The visual layer is defined in a single module (`styles.py`) so every token
lives in one place and is referenced by both CSS variables and the Plotly
theme dict.

- **Palette** — cool-neutral charcoal scale + one clay accent (`#E07856`).
  Direction (up / down) is communicated through **shape and a desaturated tint
  (sage / clay)**, not through saturated neon. No `#00FF88` / `#FF3B69`
  anywhere.
- **Typography** — Manrope (display + body, 400 → 800) + JetBrains Mono
  (numbers, labels, axis ticks). Tabular figures enforced globally.
- **Grid** — 4 / 8 / 12 / 16 / 20 / 24 / 32 / 40 / 48 / 64 px.
- **Motion** — `cubic-bezier(0.4, 0, 0.2, 1)`, 120 / 200 / 350 ms tokens.
- **Contrast** — `text-primary` on `surface-base` ≈ 16:1 · `text-muted` ≈
  5.8:1 · accent ≈ 5.1:1 — all pass WCAG AA.

## Performance

Historical 15-minute directional accuracy sits in the **55–65 % band** —
broadly consistent with academic results for short-horizon LSTMs on noisy
crypto data. Use the **Backtest on historical range** tool in the sidebar to
reproduce on any date window.

## Disclaimer

15-minute LSTM forecasts on cryptocurrency are inherently noisy. The model is
blind to news flow, macroeconomic events, and order-book pressure — its entire
input is the last 60 candles of price and momentum. Treat the output as **one
signal among many**, never as financial advice. This project exists for
educational and research purposes only.

## Author

**Ahmad Nur Fauzan** — NIM 2209106057 — Program Studi Informatika

Skripsi · 2025
