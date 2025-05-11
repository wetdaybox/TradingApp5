# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ─── Disable yfinance’s internal request cache ───────────────────────────────  
import requests_cache  
requests_cache.disabled = True  # turn off caching entirely :contentReference[oaicite:6]{index=6}  

# ─── Auto-Refresh & Page Config ────────────────────────────────────────────  
st.set_page_config(page_title="🚀 Crypto Trading Signals", layout="wide")  
st_autorefresh(interval=60_000, key="data_refresh")  # rerun every minute   

# ─── Sidebar Controls ──────────────────────────────────────────────────────  
PAIRS     = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD"]  
INTERVALS = ["5m", "60m", "1d"]  

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

# ─── Fetch Real-Time Price (reinstantiate Ticker each time) ───────────────  
def fetch_live_price(ticker_symbol: str) -> float:  
    """Recreate Ticker to bypass fast_info caching; fallback to download if needed."""  
    t = yf.Ticker(ticker_symbol)  # new instance each call :contentReference[oaicite:8]{index=8}  
    try:  
        price = float(t.fast_info.last_price)  
    except Exception:  
        data = yf.download(ticker_symbol, period="1d", interval="1m", progress=False)  
        price = float(data["Close"].iloc[-1])  # always fresh :contentReference[oaicite:9]{index=9}  
    return price  

live_price = fetch_live_price(asset)  
# Previous close via download ensures no cache  
hist_1d = yf.download(asset, period="2d", interval="1d", progress=False)  
prev_close = float(hist_1d["Close"].iloc[-2]) if len(hist_1d) >= 2 else np.nan  

# ─── Display Live Price ───────────────────────────────────────────────────  
delta = live_price - prev_close  
st.metric(f"{asset} Live Price", f"${live_price:.2f}", f"${delta:.2f}")  

# ─── Historical Data for Indicators & Backtest ────────────────────────────  
period = "60d" if interval != "1d" else "365d"  
df     = yf.download(asset, period=period, interval=interval, progress=False).dropna().copy()  
df.index = pd.to_datetime(df.index)  

# ─── Compute Indicators ───────────────────────────────────────────────────  
# RSI  
delta      = df["Close"].diff()  
gain       = delta.clip(lower=0)  
loss       = -delta.clip(upper=0)  
avg_gain   = gain.ewm(alpha=1/rsi_period, adjust=False).mean()  
avg_loss   = loss.ewm(alpha=1/rsi_period, adjust=False).mean()  
df["RSI"]  = 100 - (100 / (1 + avg_gain/avg_loss))  

# MACD  
ema_s      = df["Close"].ewm(span=macd_short, adjust=False).mean()  
ema_l      = df["Close"].ewm(span=macd_long,  adjust=False).mean()  
df["MACD"]         = ema_s - ema_l  
df["MACD_Signal"]  = df["MACD"].ewm(span=macd_signal, adjust=False).mean()  
df["MACD_Hist"]    = df["MACD"] - df["MACD_Signal"]  

# SMA  
df["SMA"] = df["Close"].rolling(window=sma_window).mean()  

# ATR for SL/TP  
hl        = df["High"] - df["Low"]  
hc        = (df["High"] - df["Close"].shift()).abs()  
lc        = (df["Low"]  - df["Close"].shift()).abs()  
df["TR"]  = hl.combine(hc, max).combine(lc, max)  
df["ATR"] = df["TR"].rolling(window=atr_period).mean()  

# ─── Edge-Triggered Signals ───────────────────────────────────────────────  
df["RSI_prev"]    = df["RSI"].shift(1)  
df["MACD_prev"]   = df["MACD_Hist"].shift(1)  

buy_rsi_cross     = (df["RSI_prev"] >= 30) & (df["RSI"] < 30)  
buy_macd_cross    = (df["MACD_prev"] <= 0) & (df["MACD_Hist"] > 0)  
sell_rsi_cross    = (df["RSI_prev"] <= 70) & (df["RSI"] > 70)  
sell_macd_cross   = (df["MACD_prev"] >= 0) & (df["MACD_Hist"] < 0)  

df["Signal"]      = "HOLD"  
df.loc[buy_rsi_cross & buy_macd_cross, "Signal"]   = "BUY"  
df.loc[sell_rsi_cross & sell_macd_cross, "Signal"] = "SELL"  

# ─── Next Action & Entry Price ───────────────────────────────────────────  
latest_signal = df["Signal"].iloc[-1]  
if latest_signal in ("BUY", "SELL"):  
    st.markdown(f"### 🚩 Signal: **{latest_signal} @ ${live_price:.2f}**")  
else:  
    st.markdown(f"### 🚩 Signal: **{latest_signal}**")  

if interval == "1d":  
    st.info("⚠️ Daily bars timestamp at 00:00 UTC. Use 5m/60m for intraday timestamps.")  

# ─── Stop-Loss & Take-Profit ─────────────────────────────────────────────  
if latest_signal in ("BUY", "SELL"):  
    atr_val = df["ATR"].iloc[-1]  
    if latest_signal == "BUY":  
        sl = live_price - 1.5 * atr_val  
        tp = live_price + 2.0 * atr_val  
    else:  
        sl = live_price + 1.5 * atr_val  
        tp = live_price - 2.0 * atr_val  
    st.markdown(  
        f"**Stop-Loss (1.5×ATR):** 🟥 ${sl:.2f}  \n"  
        f"**Take-Profit (2×ATR):** 🟩 ${tp:.2f}"  
    )  

# ─── Trade History ───────────────────────────────────────────────────────  
st.subheader("Trade History")  
hist = []  
for idx, row in df[df["Signal"] != "HOLD"].iterrows():  
    sl_hist = (row.Close - 1.5 * row.ATR) if row.Signal == "BUY" else (row.Close + 1.5 * row.ATR)  
    hist.append({  
        "Time":      idx.strftime("%Y-%m-%d %H:%M"),  
        "Signal":    row.Signal,  
        "Price":     f"${row.Close:.2f}",  
        "Stop-Loss": f"${sl_hist:.2f}"  
    })  
if hist:  
    st.table(pd.DataFrame(hist))  
else:  
    st.write("No signals in this period.")  

# ─── Plot: Price + Indicators + Signals ─────────────────────────────────  
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,  
                    row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.03,  
                    subplot_titles=("Price + SMA", "RSI", "MACD"))  

# Price + SMA  
fig.add_trace(go.Candlestick(  
    x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Candlestick"  
), row=1, col=1)  
fig.add_trace(go.Scatter(  
    x=df.index, y=df.SMA, mode="lines", name=f"SMA {sma_window}"  
), row=1, col=1)  

# Signals  
buys  = df[df.Signal == "BUY"]  
sells = df[df.Signal == "SELL"]  
fig.add_trace(go.Scatter(  
    x=buys.index, y=buys.Close, mode="markers", marker_symbol="triangle-up",  
    marker_color="green", marker_size=12, name="BUY Signal"  
), row=1, col=1)  
fig.add_trace(go.Scatter(  
    x=sells.index, y=sells.Close, mode="markers", marker_symbol="triangle-down",  
    marker_color="red", marker_size=12, name="SELL Signal"  
), row=1, col=1)  

# RSI  
fig.add_trace(go.Scatter(x=df.index, y=df.RSI, mode="lines", name="RSI"), row=2, col=1)  
fig.add_hline(y=30, line_dash="dash", line_color="red", row=2, col=1)  
fig.add_hline(y=70, line_dash="dash", line_color="green", row=2, col=1)  

# MACD  
fig.add_trace(go.Bar(x=df.index, y=df.MACD_Hist, name="MACD Hist"), row=3, col=1)  
fig.add_trace(go.Scatter(x=df.index, y=df.MACD, mode="lines", name="MACD Line"), row=3, col=1)  
fig.add_trace(go.Scatter(x=df.index, y=df.MACD_Signal, mode="lines", name="Signal Line"), row=3, col=1)  

fig.update_layout(height=800,  
                  legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),  
                  title_text=f"{asset} — Signals & Indicators")  
st.plotly_chart(fig, use_container_width=True)  

# ─── Machine Learning Prediction ─────────────────────────────────────────  
if toggle_ml:  
    st.subheader("ML Prediction")  
    ml_df = df.dropna(subset=["RSI", "MACD_Hist"]).copy()  
    ml_df["UpNext"] = (ml_df["Close"].shift(-1) > ml_df["Close"]).astype(int)  
    ml_df.dropna(inplace=True)  
    X = ml_df[["RSI", "MACD_Hist"]]; y = ml_df["UpNext"]  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
    model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)  
    model.fit(X_train, y_train)  
    acc = model.score(X_test, y_test)  
    st.write(f"Model accuracy: {acc:.2%}")  
    last_feat = pd.DataFrame({"RSI":[df.RSI.iloc[-1]], "MACD_Hist":[df.MACD_Hist.iloc[-1]]})  
    pred = model.predict(last_feat)[0]  
    st.write("Next interval:", "🔼 Up" if pred else "🔽 Down")  

# ─── Backtesting ─────────────────────────────────────────────────────────  
if toggle_bt:  
    st.subheader("Backtest Performance")  
    capital = 1000.0; position = 0; entry = 0.0; wins = 0; trades = 0  
    for sig, price in zip(df.Signal, df.Close):  
        if sig == "BUY" and position == 0:  
            position, entry, trades = 1, price, trades + 1  
        elif sig == "SELL" and position == 1:  
            pnl = price - entry; capital += pnl; wins += pnl > 0; position = 0  
    if position == 1:  
        pnl = df.Close.iloc[-1] - entry; capital += pnl; wins += pnl > 0  
    total_return = (capital - 1000.0) / 1000.0 * 100  
    win_rate      = (wins / trades * 100) if trades > 0 else 0  
    st.write(f"Total Return: **{total_return:.2f}%**, Win Rate: **{win_rate:.2f}%** ({wins}/{trades})")  
