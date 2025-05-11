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

# --- Page config & Auto-refresh ---
st.set_page_config(page_title="🚀 Crypto Trading Signals", layout="wide")
st_autorefresh(interval=60_000, key="data_refresh")  # Refresh every minute :contentReference[oaicite:0]{index=0}

# --- Sidebar ---
PAIRS     = ["BTC-USD","ETH-USD","BNB-USD","XRP-USD","ADA-USD"]  # includes XRP-USD :contentReference[oaicite:1]{index=1}
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

# --- Fetch data ---
period = "60d" if interval!="1d" else "365d"
df = yf.Ticker(asset).history(period=period, interval=interval).dropna()
df = df.copy()  # avoid chained-assignment warnings :contentReference[oaicite:2]{index=2}
df.index = pd.to_datetime(df.index)

# --- Indicators ---
# RSI (Wilder’s smoothing) :contentReference[oaicite:3]{index=3}
delta      = df["Close"].diff()
gain       = delta.clip(lower=0)
loss       = -delta.clip(upper=0)
avg_gain   = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
avg_loss   = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
df["RSI"]  = 100 - (100 / (1 + avg_gain/avg_loss))

# MACD :contentReference[oaicite:4]{index=4}
ema_s      = df["Close"].ewm(span=macd_short, adjust=False).mean()
ema_l      = df["Close"].ewm(span=macd_long,  adjust=False).mean()
df["MACD"]         = ema_s - ema_l
df["MACD_Signal"]  = df["MACD"].ewm(span=macd_signal, adjust=False).mean()
df["MACD_Hist"]    = df["MACD"] - df["MACD_Signal"]

# SMA
df["SMA"] = df["Close"].rolling(window=sma_window).mean()

# ATR for SL/TP :contentReference[oaicite:5]{index=5}
hl = df["High"] - df["Low"]
hc = (df["High"] - df["Close"].shift()).abs()
lc = (df["Low"]  - df["Close"].shift()).abs()
df["TR"]  = hl.combine(hc, max).combine(lc, max)
df["ATR"] = df["TR"].rolling(window=atr_period).mean()

# --- Crossover Logic for Signals ---
# Shifted values
df["RSI_prev"]      = df["RSI"].shift(1)
df["MACD_prev"]     = df["MACD_Hist"].shift(1)

# Edge detection
df["rsi_buy"]       = (df["RSI_prev"] >= 30) & (df["RSI"] < 30)
df["macd_buy"]      = (df["MACD_prev"] <= 0) & (df["MACD_Hist"] > 0)
df["rsi_sell"]      = (df["RSI_prev"] <= 70) & (df["RSI"] > 70)
df["macd_sell"]     = (df["MACD_prev"] >= 0) & (df["MACD_Hist"] < 0)

# Final signals: require both indicators to cross
df["Signal"]        = "HOLD"
df.loc[df["rsi_buy"] & df["macd_buy"], "Signal"]   = "BUY"
df.loc[df["rsi_sell"] & df["macd_sell"], "Signal"] = "SELL"  # only one per cross cycle :contentReference[oaicite:6]{index=6}

# --- Live Metrics & Next Action ---
current_price = float(df["Close"].iloc[-1])
prev_price    = float(df["Close"].iloc[-2])
st.metric(f"{asset} Price", f"${current_price:.2f}", f"${(current_price - prev_price):.2f}")

latest_signal = df["Signal"].iloc[-1]
if latest_signal in ("BUY","SELL"):
    st.markdown(f"### 🚩 Signal: **{latest_signal} @ ${current_price:.2f}**")  # show buy/sell price :contentReference[oaicite:7]{index=7}
else:
    st.markdown(f"### 🚩 Signal: **{latest_signal}**")

if interval=="1d":
    st.info("⚠️ Daily bars timestamp at 00:00 UTC—choose 5m/60m for real timestamps.")  # clarify midnight times :contentReference[oaicite:8]{index=8}

# --- Compute SL/TP for new trades ---
if latest_signal in ("BUY","SELL"):
    atr    = df["ATR"].iloc[-1]
    if latest_signal=="BUY":
        sl = current_price - 1.5*atr
        tp = current_price + 2.0*atr
    else:
        sl = current_price + 1.5*atr
        tp = current_price - 2.0*atr
    st.markdown(
        f"**Stop-Loss (1.5×ATR):** 🟥 ${sl:.2f}  \n"
        f"**Take-Profit (2×ATR):** 🟩 ${tp:.2f}"
    )

# --- Tidy Trade History (only real cross points) ---
st.subheader("Trade History")
hist = []
for idx, row in df[df["Signal"]!="HOLD"].iterrows():
    sl_val = row.Close - 1.5*row.ATR if row.Signal=="BUY" else row.Close + 1.5*row.ATR
    hist.append({
        "Time":       idx.strftime("%Y-%m-%d %H:%M"),
        "Signal":     row.Signal,
        "Price":      f"${row.Close:.2f}",
        "Stop-Loss":  f"${sl_val:.2f}"
    })
if hist:
    st.table(pd.DataFrame(hist))
else:
    st.write("No trades signaled in this period.")

# --- Plot: Candlestick + SMA + Signals, RSI & MACD ---
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.5,0.2,0.3], vertical_spacing=0.03,
                    subplot_titles=("Price + SMA","RSI","MACD"))

# Price & SMA
fig.add_trace(go.Candlestick(
    x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Candlestick"
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=df.SMA, mode="lines", name=f"SMA {sma_window}"
), row=1, col=1)
# Mark cross BUY/SELL
buys  = df[df.Signal=="BUY"]
sells = df[df.Signal=="SELL"]
fig.add_trace(go.Scatter(x=buys.index, y=buys.Close, mode="markers",
                         marker_symbol="triangle-up", marker_color="green", marker_size=12,
                         name="BUY Signal"), row=1, col=1)
fig.add_trace(go.Scatter(x=sells.index, y=sells.Close, mode="markers",
                         marker_symbol="triangle-down", marker_color="red", marker_size=12,
                         name="SELL Signal"), row=1, col=1)

# RSI plot
fig.add_trace(go.Scatter(x=df.index, y=df.RSI, mode="lines", name="RSI"), row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="red",    row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="green",  row=2, col=1)

# MACD plot
fig.add_trace(go.Bar(x=df.index, y=df.MACD_Hist, name="MACD Hist"), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.MACD,       mode="lines", name="MACD Line"),   row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.MACD_Signal, mode="lines", name="Signal Line"), row=3, col=1)

fig.update_layout(height=800,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  title_text=f"{asset} — Signals & Indicators")
st.plotly_chart(fig, use_container_width=True)  # interactive zoom/pan :contentReference[oaicite:9]{index=9}

# --- ML Prediction ---
if toggle_ml:
    st.subheader("ML Prediction")
    ml_df = df.dropna(subset=["RSI","MACD_Hist"]).copy()
    ml_df["NextUp"] = (ml_df["Close"].shift(-1) > ml_df["Close"]).astype(int)
    ml_df.dropna(inplace=True)
    X = ml_df[["RSI","MACD_Hist"]]; y = ml_df["NextUp"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    st.write(f"Model accuracy: {acc:.2%}")
    last_feat = pd.DataFrame({"RSI":[df.RSI.iloc[-1]], "MACD_Hist":[df.MACD_Hist.iloc[-1]]})
    pred = model.predict(last_feat)[0]
    st.write("Next interval:", "🔼 Up" if pred else "🔽 Down")

# --- Backtesting ---
if toggle_bt:
    st.subheader("Backtest Performance")
    capital=1000; position=0; entry=0; wins=0; trades=0
    for sig, price in zip(df.Signal, df.Close):
        if sig=="BUY" and position==0:
            position, entry, trades = 1, price, trades+1
        elif sig=="SELL" and position==1:
            pnl = price - entry; capital += pnl; wins += pnl>0; position=0
    if position==1:
        pnl = df.Close.iloc[-1] - entry; capital += pnl; wins += pnl>0
    ret = (capital-1000)/1000*100; wr = (wins/trades*100) if trades else 0
    st.write(f"Total Return: **{ret:.2f}%**, Win Rate: **{wr:.2f}%** ({wins}/{trades})")
