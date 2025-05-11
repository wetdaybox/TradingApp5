# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Configuration ---
st.set_page_config(page_title="🚀 Crypto Trading Signals", layout="wide")

PAIRS     = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD"]
INTERVALS = ["5m", "60m", "1d"]

# --- Sidebar ---
st.sidebar.header("Settings")
asset       = st.sidebar.selectbox("Asset", PAIRS)
interval    = st.sidebar.selectbox("Timeframe", INTERVALS, index=2)
rsi_period  = st.sidebar.slider("RSI Period", 5, 30, 14)
macd_short  = st.sidebar.number_input("MACD Short EMA", 5, 20, 12)
macd_long   = st.sidebar.number_input("MACD Long EMA", 20, 50, 26)
macd_signal = st.sidebar.number_input("MACD Signal EMA", 5, 20, 9)
sma_window  = st.sidebar.slider("SMA Window", 5, 50, 20)
atr_period  = st.sidebar.slider("ATR Period", 5, 30, 14)
toggle_ml   = st.sidebar.checkbox("ML Prediction", value=True)
toggle_bt   = st.sidebar.checkbox("Backtest",    value=True)

st.title("🚀 Crypto Trading Signal Dashboard")

# --- Fetch Data ---
period = "60d" if interval != "1d" else "max"
df = yf.Ticker(asset).history(period=period, interval=interval).dropna()
df = df.copy()

# --- Indicators ---
# RSI
delta       = df["Close"].diff()
gain        = delta.clip(lower=0)
loss        = -delta.clip(upper=0)
avg_gain    = gain.ewm(alpha=1/rsi_period).mean()
avg_loss    = loss.ewm(alpha=1/rsi_period).mean()
df["RSI"]   = 100 - (100 / (1 + avg_gain/avg_loss))
# MACD
ema_s       = df["Close"].ewm(span=macd_short, adjust=False).mean()
ema_l       = df["Close"].ewm(span=macd_long,  adjust=False).mean()
df["MACD"]        = ema_s - ema_l
df["MACD_Signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()
df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
# SMA
df["SMA"] = df["Close"].rolling(window=sma_window).mean()
# ATR for stop-loss
h_l  = df["High"] - df["Low"]
h_c  = (df["High"] - df["Close"].shift()).abs()
l_c  = (df["Low"]  - df["Close"].shift()).abs()
df["TR"]  = h_l.combine(h_c, max).combine(l_c, max)
df["ATR"] = df["TR"].rolling(window=atr_period).mean()

# --- Signal Logic ---
def gen_signal(row):
    if row.RSI < 30 and row.MACD_Hist > 0:
        return "BUY"
    if row.RSI > 70 and row.MACD_Hist < 0:
        return "SELL"
    return "HOLD"

df["Signal"] = df.apply(gen_signal, axis=1)

# --- Display Live Metrics ---
cur  = float(df["Close"].iloc[-1])
prev = float(df["Close"].iloc[-2])
st.metric(f"{asset} Price", f"${cur:.2f}", f"${cur - prev:.2f}")

# --- Display Current Signal & Stop-Loss ---
latest_sig = df["Signal"].iloc[-1]
st.markdown(f"### 🚩 Current Signal: **{latest_sig}**")

if latest_sig in ("BUY", "SELL"):
    entry = cur
    latest_atr = df["ATR"].iloc[-1]
    # For BUY, SL below; for SELL, SL above
    if latest_sig == "BUY":
        sl = entry - 1.5 * latest_atr
    else:
        sl = entry + 1.5 * latest_atr
    st.markdown(f"**Entry Price:** ${entry:.2f}  \n**Stop-Loss:** ${sl:.2f}")

# --- Annotate All Past Trades ---
st.subheader("Trade History")
history = []
for idx, row in df.iterrows():
    if row.Signal in ("BUY", "SELL"):
        # compute SL for each trade
        sl_point = (row.Close - 1.5*row.ATR) if row.Signal=="BUY" else (row.Close + 1.5*row.ATR)
        history.append({
            "Time": idx.strftime("%Y-%m-%d %H:%M"),
            "Signal": row.Signal,
            "Price": f"${row.Close:.2f}",
            "Stop-Loss": f"${sl_point:.2f}"
        })
if history:
    st.table(pd.DataFrame(history))
else:
    st.write("No trades signaled in this period.")

# --- Plot Chart ---
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.5,0.2,0.3], vertical_spacing=0.02,
    subplot_titles=("Price + SMA", "RSI", "MACD"))
# Price + SMA
fig.add_trace(go.Candlestick(
    x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close,
    name="Price"
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df.SMA, mode="lines", name=f"SMA {sma_window}"
), row=1, col=1)
# Markers for signals
buy_pts  = df[df.Signal=="BUY"]
sell_pts = df[df.Signal=="SELL"]
fig.add_trace(go.Scatter(
    x=buy_pts.index, y=buy_pts.Close, mode="markers",
    marker=dict(symbol="triangle-up",size=12,color="green"),
    name="BUY Signal"
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=sell_pts.index, y=sell_pts.Close, mode="markers",
    marker=dict(symbol="triangle-down",size=12,color="red"),
    name="SELL Signal"
), row=1, col=1)
# RSI
fig.add_trace(go.Scatter(x=df.index, y=df.RSI, name="RSI"), row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="green", row=2, col=1)
# MACD
fig.add_trace(go.Bar(x=df.index, y=df.MACD_Hist, name="MACD Hist"), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.MACD, name="MACD Line"), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.MACD_Signal, name="Signal Line"), row=3, col=1)

fig.update_layout(
    height=800,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    title_text=f"{asset} Signals & Stop-Loss"
)
st.plotly_chart(fig, use_container_width=True)

# The ML Prediction and Backtest sections remain unchanged from before...
