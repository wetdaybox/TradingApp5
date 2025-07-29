import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 s
st_autorefresh(interval=60_000, key="datarefresh")

# --- Configuration ---
BACKTEST_DAYS = 90
VOL_WINDOW = 14
VOL_MULT_MODERATE = 1.0
VOL_MULT_STRONG = 2.0
STOP_LOSS_PCT = 5.0
MAX_DRAWDOWN_PCT = 10.0

# --- Helper functions ---
@st.cache_data(ttl=300)
def fetch_data():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": "bitcoin,ripple", "vs_currencies": "usd,btc", "include_24hr_change": "true"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["bitcoin"]["usd"], data["bitcoin"]["usd_24h_change"], data["ripple"]["btc"]

@st.cache_data(ttl=600)
def fetch_historical(days: int):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    items = resp.json()["prices"]
    df = pd.DataFrame(items, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    daily = df.groupby("date").last().reset_index()
    daily["ret"] = daily["price"].pct_change()
    daily["vol"] = daily["ret"].rolling(VOL_WINDOW).std() * 100
    return daily

def compute_grid(top_price: float, drop_pct: float, levels: int):
    bottom = top_price * (1 - drop_pct / 100)
    step = (top_price - bottom) / levels
    return bottom, step

# --- App UI ---
st.set_page_config(page_title="XRP/BTC Grid Bot", layout="wide")
st.title("🟋 XRP/BTC Grid Bot with Dynamic Volatility")

# Live data
btc_price, btc_change, xrp_price = fetch_data()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("BTC/USD", f"${btc_price:,.2f}", f"{btc_change:.2f}%")
with col2:
    st.metric("XRP/BTC", f"{xrp_price:.8f} BTC")
with col3:
    st.metric("Stop‑loss", f"{STOP_LOSS_PCT:.2f}%")

# Historical volatility
daily = fetch_historical(BACKTEST_DAYS + VOL_WINDOW)
latest_vol = daily["vol"].iloc[-1]
mod_thresh = latest_vol * VOL_MULT_MODERATE
str_thresh = latest_vol * VOL_MULT_STRONG

# Sidebar config
st.sidebar.header("Parameters")
st.sidebar.write(f"14d Volatility: {latest_vol:.2f}%")
st.sidebar.write(f"Moderate threshold: {mod_thresh:.2f}%")
st.sidebar.write(f"Strong threshold: {str_thresh:.2f}%")
levels = st.sidebar.number_input("Grid levels", 1, 50, 10)
stop_loss = st.sidebar.number_input("Stop‑loss (%)", 0.0, 100.0, STOP_LOSS_PCT)
max_dd = st.sidebar.number_input("Max drawdown (%)", 0.0, 100.0, MAX_DRAWDOWN_PCT)

# Determine drop_pct
if btc_change < mod_thresh:
    drop_pct = None
elif btc_change <= str_thresh:
    drop_pct = mod_thresh
else:
    drop_pct = str_thresh

# Trigger display
if drop_pct is None:
    st.markdown(f"## No reset (BTC up {btc_change:.2f}% < {mod_thresh:.2f}% vol-thresh)")
else:
    st.markdown(f"## 🔔 Reset drop: {drop_pct:.2f}% based on volatility")

# XRP/BTC Grid
st.subheader("XRP/BTC Grid")
if drop_pct:
    bottom, step = compute_grid(xrp_price, drop_pct, levels)
    st.write(f"Top = {xrp_price:.8f} BTC | Bottom = {bottom:.8f} BTC (drop {drop_pct:.2f}%)")
    st.write(f"Step size: {step:.8f} BTC per level")
else:
    st.write("No grid adjustment.")

# Backtest button
if st.sidebar.button("Run Backtest"):
    df = daily.tail(BACKTEST_DAYS)
    trades = wins = 0
    for i in range(len(df)-1):
        if df["ret"].iloc[i]*100 >= mod_thresh:
            trades += 1
            if df["price"].iloc[i+1] > df["price"].iloc[i]:
                wins += 1
    rate = wins/trades*100 if trades else 0
    st.sidebar.write(f"Trades: {trades}, Wins: {wins}, Win Rate: {rate:.2f}%")
