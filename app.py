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

PAIRS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD"]  # Added XRP-USD
INTERVALS = ["5m", "60m", "1d"]

# --- Sidebar ---
st.sidebar.header("Controls")
asset = st.sidebar.selectbox("Asset", PAIRS)
interval = st.sidebar.selectbox("Timeframe", INTERVALS, index=2)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
macd_short = st.sidebar.number_input("MACD Short EMA", 5, 20, 12)
macd_long = st.sidebar.number_input("MACD Long EMA", 20, 50, 26)
macd_signal = st.sidebar.number_input("MACD Signal EMA", 5, 20, 9)
sma_window = st.sidebar.slider("SMA Window", 5, 50, 20)
atr_period = st.sidebar.slider("ATR Period", 5, 30, 14)  # For stop-loss / take-profit
toggle_ml = st.sidebar.checkbox("Enable ML Prediction", value=True)
toggle_backtest = st.sidebar.checkbox("Enable Backtest", value=True)

st.title("🚀 Crypto Trading Signal Dashboard")

# --- Data Fetching ---
period = "60d" if interval != "1d" else "max"
df = yf.Ticker(asset).history(period=period, interval=interval).dropna()
df.index = pd.to_datetime(df.index)

# --- Indicators ---
# RSI
delta = df["Close"].diff()
up = delta.clip(lower=0); down = -delta.clip(upper=0)
avg_gain = up.ewm(alpha=1/rsi_period).mean()
avg_loss = down.ewm(alpha=1/rsi_period).mean()
df["RSI"] = 100 - (100 / (1 + avg_gain/avg_loss))

# MACD
ema_s = df["Close"].ewm(span=macd_short, adjust=False).mean()
ema_l = df["Close"].ewm(span=macd_long,  adjust=False).mean()
df["MACD"] = ema_s - ema_l
df["MACD_Signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()
df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

# SMA
df["SMA"] = df["Close"].rolling(window=sma_window).mean()

# ATR for stop-loss/take-profit
high_low = df["High"] - df["Low"]
high_close = np.abs(df["High"] - df["Close"].shift())
low_close  = np.abs(df["Low"] - df["Close"].shift())
df["TR"] = high_low.combine(high_close, max).combine(low_close, max)
df["ATR"] = df["TR"].rolling(window=atr_period).mean()  # 1×ATR

# --- Signal Logic ---
def gen_signal(rsi, macd_hist):
    if rsi < 30 and macd_hist > 0:
        return "BUY"
    if rsi > 70 and macd_hist < 0:
        return "SELL"
    return "HOLD"

df["Signal"] = [gen_signal(r, m) for r, m in zip(df["RSI"], df["MACD_Hist"])]

# --- Live Metrics ---
# Ensure scalars for metric formatting
cur = df["Close"].iloc[-1]
prev= df["Close"].iloc[-2]
for v in ("cur","prev"):
    val = locals()[v]
    if isinstance(val, pd.Series):
        locals()[v] = val.item()

delta = cur - prev
st.metric(label=f"{asset} Price", value=f"${cur:.2f}", delta=f"{delta:.2f}")

# --- Target and Stop-Loss for Next Cycle ---
latest_atr = df["ATR"].iloc[-1]
entry_price = cur
stop_loss = entry_price - 1.5 * latest_atr      # 1.5×ATR below entry for long
take_profit = entry_price + 2.0 * latest_atr     # 2×ATR above entry for long

st.markdown(
    f"**Next Cycle Targets:**  \n"
    f"- 🟥 Stop-Loss: ${stop_loss:.2f}  \n"
    f"- 🟩 Take-Profit: ${take_profit:.2f}"
)  # ATR-based SL/TP :contentReference[oaicite:1]{index=1}

# --- Plot with Legend Key ---
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.5,0.2,0.3], vertical_spacing=0.02,
                    subplot_titles=("Price + SMA", "RSI", "MACD"))

# Price & SMA
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    name="Price (Candlestick)"
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df["SMA"], mode="lines",
    name=f"SMA {sma_window}"
), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(
    x=df.index, y=df["RSI"], mode="lines",
    name="RSI"
), row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="green", row=2, col=1)

# MACD
fig.add_trace(go.Bar(
    x=df.index, y=df["MACD_Hist"],
    name="MACD Histogram"
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df["MACD"], mode="lines",
    name="MACD Line"
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df["MACD_Signal"], mode="lines",
    name="Signal Line"
), row=3, col=1)

# Layout settings
fig.update_layout(
    height=800, 
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    title_text=f"{asset} — Signals & Indicators",
)
st.plotly_chart(fig, use_container_width=True)

# --- ML Prediction (no feature warnings) ---
if toggle_ml:
    st.subheader("ML Prediction")
    ml_df = df.dropna(subset=["RSI","MACD_Hist"])
    ml_df["Next_Close"] = ml_df["Close"].shift(-1)
    ml_df = ml_df.dropna(subset=["Next_Close"])
    ml_df["Up"] = (ml_df["Next_Close"] > ml_df["Close"]).astype(int)

    feats = ["RSI","MACD_Hist"]
    X = ml_df[feats]
    y = ml_df["Up"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    st.write(f"Test Accuracy: {acc:.2%}")

    latest_features = pd.DataFrame({feats[0]: [df["RSI"].iloc[-1]],
                                    feats[1]: [df["MACD_Hist"].iloc[-1]]})
    pred = model.predict(latest_features)[0]
    st.write("Prediction next interval:", "🔼 Up" if pred else "🔽 Down")

# --- Backtesting ---
if toggle_backtest:
    st.subheader("Backtest Results")
    capital = 1000.0; position = 0; entry = 0.0; wins = 0; trades=0
    for sig, price in zip(df["Signal"], df["Close"]):
        if sig=="BUY" and position==0:
            position, entry, trades = 1, price, trades+1
        elif sig=="SELL" and position==1:
            profit = price - entry
            capital += profit
            wins += profit>0
            position = 0
    # Close any open
    if position==1:
        profit = df["Close"].iloc[-1] - entry
        capital += profit
        wins += profit>0

    ret = (capital-1000)/1000*100
    wr  = (wins/trades*100) if trades>0 else 0
    st.write(f"Return: {ret:.2f}% • Win Rate: {wr:.2f}% ({wins}/{trades})")
