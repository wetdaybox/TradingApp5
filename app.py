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

PAIRS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD"]
INTERVALS = ["5m", "60m", "1d"]

# --- Sidebar Controls ---
st.sidebar.header("Settings")
asset        = st.sidebar.selectbox("Asset", PAIRS)
interval     = st.sidebar.selectbox("Timeframe", INTERVALS, index=2)
rsi_period   = st.sidebar.slider("RSI Period", 5, 30, 14)
macd_short   = st.sidebar.number_input("MACD Short EMA", 5, 20, 12)
macd_long    = st.sidebar.number_input("MACD Long EMA", 20, 50, 26)
macd_signal  = st.sidebar.number_input("MACD Signal EMA", 5, 20, 9)
sma_window   = st.sidebar.slider("SMA Window", 5, 50, 20)
toggle_ml    = st.sidebar.checkbox("ML Prediction", value=True)
toggle_bt    = st.sidebar.checkbox("Backtest",    value=True)

st.title("🚀 Crypto Trading Signal Dashboard")

# --- Fetch Data ---
period = "60d" if interval != "1d" else "max"
df = yf.Ticker(asset).history(period=period, interval=interval).dropna()
df = df.copy()  # avoid SettingWithCopyWarning:contentReference[oaicite:3]{index=3}

# --- Indicators ---
# RSI
delta     = df["Close"].diff()
gain      = delta.clip(lower=0)
loss      = -delta.clip(upper=0)
avg_gain  = gain.ewm(alpha=1/rsi_period).mean()
avg_loss  = loss.ewm(alpha=1/rsi_period).mean()
df["RSI"] = 100 - (100 / (1 + avg_gain/avg_loss))

# MACD
ema_s     = df["Close"].ewm(span=macd_short, adjust=False).mean()
ema_l     = df["Close"].ewm(span=macd_long,  adjust=False).mean()
df["MACD"]        = ema_s - ema_l
df["MACD_Signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()
df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

# SMA
df["SMA"] = df["Close"].rolling(window=sma_window).mean()

# --- Generate Signals ---
def gen_signal(rsi, macd_hist):
    if rsi < 30 and macd_hist > 0:
        return "BUY"   # oversold + positive momentum:contentReference[oaicite:4]{index=4}
    if rsi > 70 and macd_hist < 0:
        return "SELL"  # overbought + negative momentum
    return "HOLD"

df["Signal"] = df.apply(lambda row: gen_signal(row["RSI"], row["MACD_Hist"]), axis=1)

# --- Live Metrics ---
# Extract scalars for formatting
cur   = float(df["Close"].iloc[-1])
prev  = float(df["Close"].iloc[-2])
delta_price = cur - prev

st.metric(label=f"{asset} Price", value=f"${cur:.2f}", delta=f"${delta_price:.2f}")

# Display textual buy/sell guidance
latest_signal = df["Signal"].iloc[-1]
st.markdown(f"### 🚩 Current Signal: **{latest_signal}**")

# --- Plot Chart with Legend ---
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.5,0.2,0.3], vertical_spacing=0.03,
                    subplot_titles=("Price + SMA", "RSI", "MACD"))

# Price + SMA
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name="Price"
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df["SMA"], mode="lines", name=f"SMA {sma_window}"
), row=1, col=1)

# RSI plot
fig.add_trace(go.Scatter(
    x=df.index, y=df["RSI"], mode="lines", name="RSI"
), row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="red",    row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="green",  row=2, col=1)

# MACD plot
fig.add_trace(go.Bar(
    x=df.index, y=df["MACD_Hist"], name="MACD Hist"
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df["MACD"], mode="lines", name="MACD Line"
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df["MACD_Signal"], mode="lines", name="Signal Line"
), row=3, col=1)

fig.update_layout(
    height=800,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    title_text=f"{asset} — Signals & Indicators"
)
st.plotly_chart(fig, use_container_width=True)

# --- Optional ML Prediction ---
if toggle_ml:
    st.subheader("ML Prediction")
    ml_df = df.dropna(subset=["RSI","MACD_Hist"]).copy()
    ml_df["Next"] = (ml_df["Close"].shift(-1) > ml_df["Close"]).astype(int)
    ml_df.dropna(inplace=True)
    X = ml_df[["RSI","MACD_Hist"]]; y = ml_df["Next"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    st.write(f"Model accuracy: {acc:.2%}")
    last_feat = pd.DataFrame({"RSI":[df["RSI"].iloc[-1]], "MACD_Hist":[df["MACD_Hist"].iloc[-1]]})
    pred = model.predict(last_feat)[0]
    st.write("Next move:", "🔼 **Up**" if pred else "🔽 **Down**")

# --- Optional Backtest ---
if toggle_bt:
    st.subheader("Backtest Results")
    capital=1000; position=0; entry=0; wins=0; trades=0
    for sig, price in zip(df["Signal"], df["Close"]):
        if sig=="BUY" and position==0:
            position, entry, trades = 1, price, trades+1
        elif sig=="SELL" and position==1:
            pnl = price - entry
            capital += pnl
            wins += pnl>0
            position = 0
    if position==1:
        pnl = df["Close"].iloc[-1] - entry
        capital += pnl; wins += pnl>0
    ret = (capital-1000)/1000*100
    wr  = wins/trades*100 if trades>0 else 0
    st.write(f"Return: **{ret:.2f}%**, Win rate: **{wr:.2f}%** ({wins}/{trades})")
