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

# ─── Disable yfinance’s built-in HTTP cache ────────────────────────────────
import requests_cache
requests_cache.disabled = True  # turn off all caching :contentReference[oaicite:3]{index=3}

# ─── Auto-refresh & page config ───────────────────────────────────────────
st.set_page_config(page_title="🚀 Crypto Trading Signals", layout="wide")
st_autorefresh(interval=60_000, key="data_refresh")  # refresh every minute 

# ─── Sidebar controls ─────────────────────────────────────────────────────
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

# ─── Helper to fetch a truly live price ────────────────────────────────────
def fetch_live_price(symbol: str) -> float:
    """Recreate Ticker to avoid cached fast_info; fallback to download if needed."""
    t = yf.Ticker(symbol)
    try:
        # fast_info is very fast but cached per Ticker instance :contentReference[oaicite:5]{index=5}
        return float(t.fast_info.last_price)
    except Exception:
        # yf.download always fetches fresh OHLC bars :contentReference[oaicite:6]{index=6}
        df = yf.download(symbol, period="1d", interval="1m", progress=False)
        return float(df["Close"].iloc[-1])

live_price = fetch_live_price(asset)

# ─── Previous close for delta ─────────────────────────────────────────────
hist_d = yf.download(asset, period="2d", interval="1d", progress=False)
prev_close = float(hist_d["Close"].iloc[-2]) if len(hist_d) >= 2 else np.nan

# ─── Display live metric ─────────────────────────────────────────────────
st.metric(f"{asset} Live Price", f"${live_price:.2f}", f"${(live_price - prev_close):.2f}")

# ─── Historical data for indicators and backtest ─────────────────────────
period = "60d" if interval != "1d" else "365d"
df     = yf.download(asset, period=period, interval=interval, progress=False).dropna().copy()
df.index = pd.to_datetime(df.index)

# ─── Compute RSI ─────────────────────────────────────────────────────────
delta     = df["Close"].diff()
gain      = delta.clip(lower=0)
loss      = -delta.clip(upper=0)
avg_gain  = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
avg_loss  = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
df["RSI"] = 100 - (100 / (1 + avg_gain/avg_loss))

# ─── Compute MACD ───────────────────────────────────────────────────────
ema_s      = df["Close"].ewm(span=macd_short, adjust=False).mean()
ema_l      = df["Close"].ewm(span=macd_long,  adjust=False).mean()
df["MACD"]         = ema_s - ema_l
df["MACD_Signal"]  = df["MACD"].ewm(span=macd_signal, adjust=False).mean()
df["MACD_Hist"]    = df["MACD"] - df["MACD_Signal"]

# ─── Compute SMA & ATR ───────────────────────────────────────────────────
df["SMA"] = df["Close"].rolling(window=sma_window).mean()
hl        = df["High"] - df["Low"]
hc        = (df["High"] - df["Close"].shift()).abs()
lc        = (df["Low"]  - df["Close"].shift()).abs()
df["TR"]  = hl.combine(hc, max).combine(lc, max)
df["ATR"] = df["TR"].rolling(window=atr_period).mean()

# ─── Edge-triggered Signal Logic ─────────────────────────────────────────
df["RSI_prev"]   = df["RSI"].shift(1)
df["MACD_prev"]  = df["MACD_Hist"].shift(1)
buy_rsi = (df["RSI_prev"] >= 30) & (df["RSI"] < 30)
buy_macd= (df["MACD_prev"] <= 0) & (df["MACD_Hist"] > 0)
sell_rsi= (df["RSI_prev"] <= 70) & (df["RSI"] > 70)
sell_macd=(df["MACD_prev"] >= 0) & (df["MACD_Hist"] < 0)

df["Signal"] = "HOLD"
df.loc[buy_rsi & buy_macd,   "Signal"] = "BUY"
df.loc[sell_rsi & sell_macd, "Signal"] = "SELL"

# ─── Next Action & Entry Price ───────────────────────────────────────────
latest_signal = df["Signal"].iloc[-1]
if latest_signal in ("BUY", "SELL"):
    st.markdown(f"### 🚩 Signal: **{latest_signal} @ ${live_price:.2f}**")
else:
    st.markdown(f"### 🚩 Signal: **{latest_signal}**")

if interval == "1d":
    st.info("⚠️ Daily bars timestamp at 00:00 UTC; use 5m/60m for intraday updates.")

# ─── Stop-Loss & Take-Profit ─────────────────────────────────────────────
if latest_signal in ("BUY", "SELL"):
    atr_val = df["ATR"].iloc[-1]
    if latest_signal == "BUY":
        sl = live_price - 1.5*atr_val
        tp = live_price + 2.0*atr_val
    else:
        sl = live_price + 1.5*atr_val
        tp = live_price - 2.0*atr_val
    st.markdown(
        f"**Stop-Loss (1.5×ATR):** 🟥 ${sl:.2f}  \n"
        f"**Take-Profit (2×ATR):** 🟩 ${tp:.2f}"
    )

# ─── Trade History ───────────────────────────────────────────────────────
st.subheader("Trade History")
history = []
for ts, row in df[df["Signal"]!="HOLD"].iterrows():
    sl_val = row.Close - 1.5*row.ATR if row.Signal=="BUY" else row.Close + 1.5*row.ATR
    history.append({
        "Time":      ts.strftime("%Y-%m-%d %H:%M"),
        "Signal":    row.Signal,
        "Price":     f"${row.Close:.2f}",
        "Stop-Loss": f"${sl_val:.2f}"
    })
if history:
    st.table(pd.DataFrame(history))
else:
    st.write("No signals in this period.")

# ─── Plot Price, Indicators & Signals ────────────────────────────────────
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.5,0.2,0.3], vertical_spacing=0.03,
                    subplot_titles=("Price + SMA","RSI","MACD"))

fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High,
                             low=df.Low, close=df.Close, name="Candlestick"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.SMA, mode="lines", name=f"SMA {sma_window}"), row=1, col=1)
buys  = df[df.Signal=="BUY"];   sells = df[df.Signal=="SELL"]
fig.add_trace(go.Scatter(x=buys.index, y=buys.Close, mode="markers",
                         marker_symbol="triangle-up", marker_color="green", name="BUY Signal"), row=1, col=1)
fig.add_trace(go.Scatter(x=sells.index, y=sells.Close, mode="markers",
                         marker_symbol="triangle-down", marker_color="red",   name="SELL Signal"), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df.RSI, mode="lines", name="RSI"), row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="red",   row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="green", row=2, col=1)

fig.add_trace(go.Bar(x=df.index, y=df.MACD_Hist, name="MACD Hist"), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.MACD, name="MACD Line"), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.MACD_Signal, name="Signal Line"), row=3, col=1)

fig.update_layout(height=800, legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
st.plotly_chart(fig, use_container_width=True)

# ─── ML Prediction & Backtest (unchanged) ────────────────────────────────
# ... (your existing ML and backtest code here) ...
