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

PAIRS     = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD"]  # Includes XRP-USD :contentReference[oaicite:0]{index=0}
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
atr_period  = st.sidebar.slider("ATR Period", 5, 30, 14)  # For SL/TP  
toggle_ml   = st.sidebar.checkbox("Enable ML Prediction", value=True)
toggle_bt   = st.sidebar.checkbox("Enable Backtest",    value=True)

st.title("🚀 Crypto Trading Signal Dashboard")

# --- Fetch & Prepare Data ---
period = "60d" if interval != "1d" else "max"
df = yf.Ticker(asset).history(period=period, interval=interval).dropna()
df = df.copy()  # avoids SettingWithCopyWarning :contentReference[oaicite:1]{index=1}

# --- Compute Indicators ---
# RSI (Wilder’s method) :contentReference[oaicite:2]{index=2}
delta       = df["Close"].diff()
gain        = delta.clip(lower=0)
loss        = -delta.clip(upper=0)
avg_gain    = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
avg_loss    = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
df["RSI"]   = 100 - (100 / (1 + avg_gain/avg_loss))

# MACD & Signal Line :contentReference[oaicite:3]{index=3}
ema_s       = df["Close"].ewm(span=macd_short, adjust=False).mean()
ema_l       = df["Close"].ewm(span=macd_long,  adjust=False).mean()
df["MACD"]        = ema_s - ema_l
df["MACD_Signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()
df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

# SMA
df["SMA"] = df["Close"].rolling(window=sma_window).mean()

# ATR for Stop-Loss/Take-Profit :contentReference[oaicite:4]{index=4}
hl = df["High"] - df["Low"]
hc = (df["High"] - df["Close"].shift()).abs()
lc = (df["Low"]  - df["Close"].shift()).abs()
df["TR"]  = hl.combine(hc, max).combine(lc, max)
df["ATR"] = df["TR"].rolling(window=atr_period).mean()

# --- Generate Signals ---
def gen_signal(row):
    # Classic RSI + MACD histogram logic
    if row.RSI < 30 and row.MACD_Hist > 0:
        return "BUY"
    if row.RSI > 70 and row.MACD_Hist < 0:
        return "SELL"
    return "HOLD"

df["Signal"] = df.apply(gen_signal, axis=1)

# --- Live Price & Next Action ---
current_price = float(df["Close"].iloc[-1])
previous_price= float(df["Close"].iloc[-2])
st.metric(f"{asset} Price", f"${current_price:.2f}", f"${(current_price-previous_price):.2f}")

latest_signal = df["Signal"].iloc[-1]
st.markdown(f"### 🚩 Next Action: **{latest_signal}**")

# Compute SL & TP for a trade if signal is BUY or SELL
if latest_signal in ("BUY","SELL"):
    atr_value = df["ATR"].iloc[-1]
    if latest_signal == "BUY":
        sl = current_price - 1.5 * atr_value
        tp = current_price + 2.0 * atr_value
    else:
        sl = current_price + 1.5 * atr_value
        tp = current_price - 2.0 * atr_value
    st.markdown(
        f"**Stop-Loss:** 🟥 ${sl:.2f}  \n"
        f"**Take-Profit:** 🟩 ${tp:.2f}"
    )

# --- Tidy Trade History ---
st.subheader("Trade Signals History")
hist = []
for ts, row in df.iterrows():
    if row.Signal in ("BUY","SELL"):
        sl_hist = (row.Close - 1.5*row.ATR) if row.Signal=="BUY" else (row.Close + 1.5*row.ATR)
        hist.append((ts.strftime("%Y-%m-%d %H:%M"),
                     row.Signal,
                     f"${row.Close:.2f}",
                     f"${sl_hist:.2f}"))
if hist:
    hist_df = pd.DataFrame(hist, columns=["Time","Signal","Price","Stop-Loss"])
    st.table(hist_df)
else:
    st.write("No signals in history for this timeframe.")

# --- Plot Interactive Chart with Legend Key :contentReference[oaicite:5]{index=5}---
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.5,0.2,0.3], vertical_spacing=0.03,
                    subplot_titles=("Price + SMA","RSI","MACD"))

# Price + SMA traces
fig.add_trace(go.Candlestick(
    x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Candlestick"
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df.SMA, mode="lines", name=f"SMA {sma_window}"
), row=1, col=1)
# Signal markers
buys  = df[df.Signal=="BUY"]
sells = df[df.Signal=="SELL"]
fig.add_trace(go.Scatter(
    x=buys.index, y=buys.Close, mode="markers", marker_symbol="triangle-up",
    marker_color="green", marker_size=12, name="BUY Signal"
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=sells.index, y=sells.Close, mode="markers", marker_symbol="triangle-down",
    marker_color="red", marker_size=12, name="SELL Signal"
), row=1, col=1)

# RSI subplot
fig.add_trace(go.Scatter(x=df.index, y=df.RSI, mode="lines", name="RSI"), row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="red",    row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="green",  row=2, col=1)

# MACD subplot
fig.add_trace(go.Bar(x=df.index, y=df.MACD_Hist, name="MACD Hist"), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.MACD, mode="lines", name="MACD Line"), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.MACD_Signal, mode="lines", name="MACD Signal"), row=3, col=1)

fig.update_layout(
    height=800,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    title_text=f"{asset} Signals & Indicators"
)
st.plotly_chart(fig, use_container_width=True)

# --- Machine Learning Prediction ---
if toggle_ml:
    st.subheader("ML Direction Prediction")
    ml_df = df.dropna(subset=["RSI","MACD_Hist","Close"]).copy()
    ml_df["NextUp"] = (ml_df["Close"].shift(-1) > ml_df["Close"]).astype(int)
    ml_df.dropna(inplace=True)
    X = ml_df[["RSI","MACD_Hist"]]; y = ml_df["NextUp"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    st.write(f"Model Accuracy: {score:.2%}")
    last_features = pd.DataFrame({
        "RSI":[df["RSI"].iloc[-1]],
        "MACD_Hist":[df["MACD_Hist"].iloc[-1]]
    })
    pred = model.predict(last_features)[0]
    st.write("Next Interval Prediction:", "🔼 Up" if pred else "🔽 Down")

# --- Backtesting Module ---
if toggle_bt:
    st.subheader("Backtest Performance")
    capital = 1000.0; position = 0; entry=0.0; wins=0; trades=0
    for sig, price in zip(df.Signal, df.Close):
        if sig=="BUY" and position==0:
            position, entry, trades = 1, price, trades+1
        elif sig=="SELL" and position==1:
            pnl = price - entry
            capital += pnl
            wins += pnl>0
            position = 0
    # close last
    if position==1:
        pnl = df.Close.iloc[-1] - entry
        capital += pnl; wins += pnl>0
    ret = (capital-1000)/1000*100
    wr  = wins/trades*100 if trades>0 else 0
    st.write(f"Total Return: **{ret:.2f}%**, Win Rate: **{wr:.2f}%** ({wins}/{trades})")
