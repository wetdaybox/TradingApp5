# app.py
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ─── Page Config & Auto-Refresh ───────────────────────────────────────────
st.set_page_config(page_title="🚀 Crypto Trading Signals", layout="wide")
st_autorefresh(interval=60_000, key="data_refresh")  # rerun every minute :contentReference[oaicite:3]{index=3}

# ─── Sidebar Controls ─────────────────────────────────────────────────────
PAIRS     = ["BTC-USD","ETH-USD","BNB-USD","XRP-USD","ADA-USD"]
INTERVALS = ["5m","60m","1d"]
st.sidebar.header("Settings")
asset       = st.sidebar.selectbox("Asset", PAIRS)
interval    = st.sidebar.selectbox("Interval", INTERVALS, index=0)
rsi_period  = st.sidebar.slider("RSI Period", 5, 30, 14)
macd_short  = st.sidebar.number_input("MACD Short EMA", 5, 20, 12)
macd_long   = st.sidebar.number_input("MACD Long EMA", 20, 50, 26)
macd_signal = st.sidebar.number_input("MACD Signal EMA", 5, 20, 9)
sma_window  = st.sidebar.slider("SMA Window", 5, 50, 20)
atr_period  = st.sidebar.slider("ATR Period", 5, 30, 14)
toggle_ml   = st.sidebar.checkbox("Enable ML Prediction", value=True)
toggle_bt   = st.sidebar.checkbox("Enable Backtest",    value=True)

st.title("🚀 Crypto Trading Signal Dashboard")

# ─── Caching Historical Data ──────────────────────────────────────────────
@st.cache_data(ttl=300)  # cache for 5 minutes :contentReference[oaicite:4]{index=4}
def get_historical(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

# ─── Live Price Fetch (always fresh) ─────────────────────────────────────
def fetch_live_price(symbol: str) -> float:
    t = yf.Ticker(symbol)  # fresh Ticker to bypass cache :contentReference[oaicite:5]{index=5}
    try:
        return float(t.fast_info.last_price)
    except:
        df1m = yf.download(symbol, period="1d", interval="1m", progress=False)
        return float(df1m["Close"].iloc[-1])

live_price = fetch_live_price(asset)

# ─── Previous Close for Delta ─────────────────────────────────────────────
hist2d = get_historical(asset, "2d", "1d")
# use .iloc[0] to avoid FutureWarning :contentReference[oaicite:6]{index=6}
prev_close = float(hist2d["Close"].iloc[-2:].iloc[0]) if len(hist2d) >= 2 else np.nan

st.metric(f"{asset} Live Price", f"${live_price:.2f}", f"${live_price - prev_close:.2f}")

# ─── Fetch & Display Historical Data ─────────────────────────────────────
period = "60d" if interval != "1d" else "365d"
with st.spinner("Loading historical data…"):  # indicate progress :contentReference[oaicite:7]{index=7}
    df = get_historical(asset, period, interval)

# ─── Technical Indicators ─────────────────────────────────────────────────
# RSI
delta     = df.Close.diff()
gain      = delta.clip(lower=0)
loss      = -delta.clip(upper=0)
avg_gain  = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
avg_loss  = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
df["RSI"] = 100 - (100/(1 + avg_gain/avg_loss))

# MACD
ema_s      = df.Close.ewm(span=macd_short, adjust=False).mean()
ema_l      = df.Close.ewm(span=macd_long,  adjust=False).mean()
df["MACD"]        = ema_s - ema_l
df["MACD_Signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()
df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

# SMA
df["SMA"] = df.Close.rolling(window=sma_window).mean()

# True Range & ATR (row-wise max) :contentReference[oaicite:8]{index=8}
hl        = df.High - df.Low
hc        = (df.High - df.Close.shift()).abs()
lc        = (df.Low  - df.Close.shift()).abs()
df["TR"]  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
df["ATR"] = df["TR"].rolling(window=atr_period).mean()

# ─── Generate Signals ────────────────────────────────────────────────────
df["RSI_prev"]  = df.RSI.shift(1)
df["MACD_prev"] = df.MACD_Hist.shift(1)
buy_rsi   = (df.RSI_prev >= 30) & (df.RSI < 30)
buy_macd  = (df.MACD_prev <= 0) & (df.MACD_Hist > 0)
sell_rsi  = (df.RSI_prev <= 70) & (df.RSI > 70)
sell_macd = (df.MACD_prev >= 0) & (df.MACD_Hist < 0)
df["Signal"] = "HOLD"
df.loc[buy_rsi & buy_macd,   "Signal"] = "BUY"
df.loc[sell_rsi & sell_macd, "Signal"] = "SELL"

# ─── Display Latest Signal ────────────────────────────────────────────────
latest = df.Signal.iloc[-1]
if latest in ("BUY","SELL"):
    st.markdown(f"### 🚩 Signal: **{latest} @ ${live_price:.2f}**")
else:
    st.markdown(f"### 🚩 Signal: **{latest}**")
if interval == "1d":
    st.info("⚠️ Daily bars timestamp at 00:00 UTC; use 5m/60m for intraday updates.")

# ─── ATR-based Stop-Loss & Take-Profit ────────────────────────────────────
if latest in ("BUY","SELL"):
    atrv = df.ATR.iloc[-1]
    sl   = live_price - 1.5*atrv if latest=="BUY" else live_price + 1.5*atrv
    tp   = live_price + 2.0*atrv if latest=="BUY" else live_price - 2.0*atrv
    st.markdown(
        f"**Stop-Loss (1.5×ATR):** 🟥 ${sl:.2f}  \n"
        f"**Take-Profit (2×ATR):** 🟩 ${tp:.2f}"
    )

# ─── Trade History ────────────────────────────────────────────────────────
st.subheader("Trade History")
history = []
for ts, signal, close, atr in df.loc[df.Signal!="HOLD", ["Signal","Close","ATR"]].itertuples(index=True, name=None):
    slh = close - 1.5*atr if signal=="BUY" else close + 1.5*atr
    history.append({
        "Time":      pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M"),
        "Signal":    signal,
        "Price":     f"${close:.2f}",
        "Stop-Loss": f"${slh:.2f}"
    })
if history:
    st.table(pd.DataFrame(history))
else:
    st.write("No signals in this period.")

# ─── Plot & ML & Backtest (omitted for brevity)—unchanged ────────────────
# (Include your existing Plotly, ML prediction, and backtesting blocks here.)

