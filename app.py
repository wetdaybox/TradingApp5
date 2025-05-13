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
st_autorefresh(interval=60_000, key="data_refresh")  # rerun every 60s :contentReference[oaicite:3]{index=3}

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

# ─── Cache Only the Download (with TTL) ───────────────────────────────────
@st.cache_data(ttl=300)  # cache for 5 minutes :contentReference[oaicite:4]{index=4}
def download_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

# ─── Live Price Fetch (always fresh) ─────────────────────────────────────
def fetch_live_price(symbol: str) -> float:
    t = yf.Ticker(symbol)  # fresh Ticker instance :contentReference[oaicite:5]{index=5}
    try:
        return float(t.fast_info.last_price)
    except:
        df1m = yf.download(symbol, period="1d", interval="1m", progress=False)
        return float(df1m["Close"].iloc[-1])

live_price = fetch_live_price(asset)

# ─── Previous Close for Delta ─────────────────────────────────────────────
hist_2d = download_data(asset, "2d", "1d")
# Use .iloc[0] to avoid FutureWarning :contentReference[oaicite:6]{index=6}
prev_close = float(hist_2d["Close"].iloc[-2:].iloc[0]) if len(hist_2d) >= 2 else np.nan

st.metric(f"{asset} Live Price", f"${live_price:.2f}", f"${live_price - prev_close:.2f}")

# ─── Download Historical for Indicators & Backtest ────────────────────────
period = "60d" if interval != "1d" else "365d"
with st.spinner("Loading historical data…"):  # show spinner during slow download :contentReference[oaicite:7]{index=7}
    df = download_data(asset, period, interval)

# ─── Compute Technical Indicators ─────────────────────────────────────────
# RSI
delta     = df["Close"].diff()
gain      = delta.clip(lower=0)
loss      = -delta.clip(upper=0)
avg_gain  = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
avg_loss  = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
df["RSI"] = 100 - (100/(1 + avg_gain/avg_loss))

# MACD
ema_s      = df["Close"].ewm(span=macd_short, adjust=False).mean()
ema_l      = df["Close"].ewm(span=macd_long,  adjust=False).mean()
df["MACD"]        = ema_s - ema_l
df["MACD_Signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()
df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

# SMA
df["SMA"] = df["Close"].rolling(window=sma_window).mean()

# True Range & ATR (element-wise max) :contentReference[oaicite:8]{index=8}
hl = df["High"] - df["Low"]
hc = (df["High"] - df["Close"].shift()).abs()
lc = (df["Low"]  - df["Close"].shift()).abs()
df["TR"]  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
df["ATR"] = df["TR"].rolling(window=atr_period).mean()

# ─── Generate Edge-Triggered Signals ─────────────────────────────────────
df["RSI_prev"]  = df["RSI"].shift(1)
df["MACD_prev"] = df["MACD_Hist"].shift(1)
buy_rsi   = (df["RSI_prev"] >= 30) & (df["RSI"] < 30)
buy_macd  = (df["MACD_prev"] <= 0) & (df["MACD_Hist"] > 0)
sell_rsi  = (df["RSI_prev"] <= 70) & (df["RSI"] > 70)
sell_macd = (df["MACD_prev"] >= 0) & (df["MACD_Hist"] < 0)
df["Signal"] = "HOLD"
df.loc[buy_rsi & buy_macd,   "Signal"] = "BUY"
df.loc[sell_rsi & sell_macd, "Signal"] = "SELL"

# ─── Display Latest Signal & ATR-based SL/TP ─────────────────────────────
latest = df["Signal"].iloc[-1]
if latest in ("BUY", "SELL"):
    st.markdown(f"### 🚩 Signal: **{latest} @ ${live_price:.2f}**")
else:
    st.markdown(f"### 🚩 Signal: **{latest}**")

if interval == "1d":
    st.info("⚠️ Daily bars timestamp at 00:00 UTC; use 5m/60m for intraday.")

if latest in ("BUY", "SELL"):
    atrv = df["ATR"].iloc[-1]
    sl   = live_price - 1.5*atrv if latest=="BUY" else live_price + 1.5*atrv
    tp   = live_price + 2.0*atrv if latest=="BUY" else live_price - 2.0*atrv
    st.markdown(f"**Stop-Loss:** ${sl:.2f}  \n**Take-Profit:** ${tp:.2f}")

# ─── Trade History (scalar-safe) ─────────────────────────────────────────
st.subheader("Trade History")
history = []
for ts, signal, close, atr in df.loc[df["Signal"]!="HOLD", ["Signal","Close","ATR"]].itertuples(index=True, name=None):
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

# ─── Plot Price, Indicators & Signals ────────────────────────────────────
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
# Signals
buys  = df[df.Signal=="BUY"]; sells = df[df.Signal]=="SELL"
fig.add_trace(go.Scatter(x=buys.index, y=buys.Close, mode="markers",
                         marker_symbol="triangle-up", marker_color="green", name="BUY"), row=1, col=1)
fig.add_trace(go.Scatter(x=sells.index,y=sells.Close,mode="markers",
                         marker_symbol="triangle-down",marker_color="red",   name="SELL"), row=1, col=1)
# RSI
fig.add_trace(go.Scatter(x=df.index, y=df.RSI, mode="lines", name="RSI"), row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="red",   row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="green", row=2, col=1)
# MACD
fig.add_trace(go.Bar(x=df.index, y=df.MACD_Hist, name="MACD Hist"), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.MACD,       mode="lines", name="MACD Line"),   row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.MACD_Signal, mode="lines", name="Signal Line"), row=3, col=1)
fig.update_layout(height=800, legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
st.plotly_chart(fig, use_container_width=True)

# ─── ML Prediction ────────────────────────────────────────────────────────
if toggle_ml and {"RSI","MACD_Hist"}.issubset(df.columns):
    st.subheader("ML Prediction")
    ml_df = df.dropna(subset=["RSI","MACD_Hist"]).copy()
    if not ml_df.empty:
        ml_df["UpNext"] = (ml_df.Close.shift(-1) > ml_df.Close).astype(int)
        ml_df.dropna(inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(
            ml_df[["RSI","MACD_Hist"]], ml_df["UpNext"], test_size=0.2, random_state=42
        )
        model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
        model.fit(X_train, y_train)
        st.write(f"Model accuracy: {model.score(X_test, y_test):.2%}")
        last_feat = pd.DataFrame({"RSI":[df.RSI.iloc[-1]], "MACD_Hist":[df.MACD_Hist.iloc[-1]]})
        pred = model.predict(last_feat)[0]
        st.write("Next interval:", "🔼 Up" if pred else "🔽 Down")
    else:
        st.write("Insufficient data for ML prediction.")
else:
    st.write("ML prediction disabled or missing indicators.")

# ─── Backtesting ────────────────────────────────────────────────────────
if toggle_bt:
    st.subheader("Backtest Performance")
    capital, position, entry, wins, trades = 1000, 0, 0, 0, 0
    for sig, price in zip(df.Signal, df.Close):
        if sig=="BUY" and position==0:
            position, entry, trades = 1, price, trades+1
        elif sig=="SELL" and position==1:
            pnl = price - entry; capital += pnl; wins += pnl>0; position=0
    if position==1:
        pnl = df.Close.iloc[-1] - entry; capital += pnl; wins += pnl>0
    total_return = (capital - 1000)/1000*100
    win_rate      = (wins/trades*100) if trades>0 else 0
    st.write(f"Total Return: **{total_return:.2f}%**, Win Rate: **{win_rate:.2f}%** ({wins}/{trades})")
