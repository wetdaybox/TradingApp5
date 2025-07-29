import streamlit as st
import requests
import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 seconds
st_autorefresh(interval=60_000, key="datarefresh")

# Configuration
HISTORY_DAYS = 365
VOL_WINDOW = 14
RSI_WINDOW = 14
SMA_SHORT = 5
SMA_LONG = 20
EMA_TREND = 200
RSI_OVERBOUGHT = 75
GRID_LEVELS = 10

BINANCE = "https://api.binance.com/api/v3"

# Fetch live BTC/USD and XRP/BTC from Binance; fallback to CoinGecko if needed
@st.cache_data(ttl=60)
def fetch_live():
    try:
        r = requests.get(f"{BINANCE}/ticker/24hr", params={"symbol": "BTCUSDT"}, timeout=5)
        r.raise_for_status()
        bd = r.json()
        btc_price = float(bd["lastPrice"])
        btc_change = float(bd["priceChangePercent"])
        r = requests.get(f"{BINANCE}/ticker/24hr", params={"symbol": "XRPBTC"}, timeout=5)
        r.raise_for_status()
        xd = r.json()
        xrp_price = float(xd["lastPrice"])
        return btc_price, btc_change, xrp_price
    except Exception:
        # fallback to CoinGecko
        params = {"ids":"bitcoin,ripple","vs_currencies":"usd,btc","include_24hr_change":"true"}
        r = requests.get("https://api.coingecko.com/api/v3/simple/price", params=params, timeout=10)
        r.raise_for_status()
        d = r.json()
        return (
            d["bitcoin"]["usd"],
            d["bitcoin"]["usd_24h_change"],
            d["ripple"]["btc"]
        )

# Fetch last HISTORY_DAYS+EMA_TREND days of BTC daily closes
@st.cache_data(ttl=600)
def fetch_history():
    limit = HISTORY_DAYS + EMA_TREND
    r = requests.get(f"{BINANCE}/klines", params={
        "symbol":"BTCUSDT","interval":"1d","limit":limit}, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "openTime","open","high","low","close","vol","closeTime","qVol","trades","takerB","takerQ","ignore"
    ])
    df["date"] = pd.to_datetime(df["openTime"], unit="ms")
    df = df.set_index("date")
    df["price"] = df["close"].astype(float)
    df["ret"] = df["price"].pct_change()*100
    # Indicators
    df["vol14"] = df["ret"].rolling(VOL_WINDOW).std()
    df["sma5"] = df["price"].rolling(SMA_SHORT).mean()
    df["sma20"] = df["price"].rolling(SMA_LONG).mean()
    df["ema200"] = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    delta = df["price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    df["rsi"] = 100 - 100/(1 + gain.rolling(RSI_WINDOW).mean()/loss.rolling(RSI_WINDOW).mean())
    return df.dropna()

# Compute grid bottom & step
def compute_grid(top, pct, levels):
    bottom = top*(1 - pct/100)
    step = (top - bottom)/levels
    return bottom, step

# Main app
st.set_page_config(page_title="XRP/BTC Grid Bot", layout="wide")
st.title("🟋 XRP/BTC Grid Bot with Dynamic Thresholds")

# 1. Fetch live
btc_price, btc_change, xrp_price = fetch_live()

# 2. Fetch history & indicators
hist = fetch_history()
row = hist.iloc[-1]

# 3. Build dynamic thresholds
mod_th = row["vol14"]            # 1×14‑day vol
str_th = 2*row["vol14"]          # 2×14‑day vol

# 4. Confirmation filters
regime_ok   = row["price"] > row["ema200"]
momentum_ok = row["sma5"] > row["sma20"]
rsi_ok      = row["rsi"] < RSI_OVERBOUGHT

# 5. Trigger logic
if btc_change < mod_th:
    drop_pct = None
    status = f"No reset (BTC up {btc_change:.2f}% < {mod_th:.2f}%)"
elif btc_change <= str_th:
    drop_pct = mod_th
    status = f"🔔 Moderate reset: drop {mod_th:.2f}%"
else:
    drop_pct = str_th
    status = f"🔔 Strong reset: drop {str_th:.2f}%"

trigger = drop_pct is not None and regime_ok and momentum_ok and rsi_ok

# Display live metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC/USD", f"${btc_price:,.2f}", f"{btc_change:.2f}%")
c2.metric("XRP/BTC", f"{xrp_price:.8f} BTC")
c3.metric("14d Volatility", f"{row['vol14']:.2f}%")
c4.metric("RSI/EMA Filters", f"{'OK' if regime_ok and momentum_ok and rsi_ok else 'NO'}")

st.markdown(f"## {status}  |  Filters: Regime={'OK' if regime_ok else 'NO'}, "
            f"Momentum={'OK' if momentum_ok else 'NO'}, RSI={'OK' if rsi_ok else 'NO'}")

# Sidebar grid levels
levels = st.sidebar.number_input("Grid levels", 1, 50, 10)

# Calculate grid
if trigger:
    bottom, step = compute_grid(xrp_price, drop_pct, levels)
    st.write(f"**Grid top:** {xrp_price:.8f} BTC | **bottom:** {bottom:.8f} BTC")
    st.write(f"**Step size:** {step:.8f} BTC over {levels} levels")
else:
    st.write("No grid adjustment at this time.")

# Requirements note
st.write("---")
st.write("Requirements: streamlit, requests, pandas, numpy, streamlit-autorefresh")
