import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Auto‑refresh every 60 s
st_autorefresh(interval=60_000, key="datarefresh")

# ── CONFIGURATION ──
HISTORY_DAYS = 90          # <= 90 so market_chart works reliably
VOL_WINDOW = 14            # for volatility
RSI_WINDOW = 14
SMA_SHORT = 5
SMA_LONG = 20
EMA_TREND = 50             # adjusted to 50 days
RSI_OVERBOUGHT = 75
GRID_LEVELS_DEFAULT = 10

# ── FETCH LIVE DATA ──
@st.cache_data(ttl=60)
def fetch_live():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin,ripple",
        "vs_currencies": "usd,btc",
        "include_24hr_change": "true"
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    d = r.json()
    return (
        d["bitcoin"]["usd"],
        d["bitcoin"]["usd_24h_change"],
        d["ripple"]["btc"]
    )

# ── FETCH HISTORICAL DATA ──
@st.cache_data(ttl=600)
def fetch_history(days):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    prices = r.json()["prices"]
    df = pd.DataFrame(prices, columns=["ts", "price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change() * 100
    # Indicators
    df[f"vol{VOL_WINDOW}"] = df["return"].rolling(VOL_WINDOW).std()
    df["sma_short"] = df["price"].rolling(SMA_SHORT).mean()
    df["sma_long"] = df["price"].rolling(SMA_LONG).mean()
    df["ema_trend"] = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    # RSI
    delta = df["price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_WINDOW).mean()
    avg_loss = loss.rolling(RSI_WINDOW).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - 100 / (1 + rs)
    return df.dropna()

# ── GRID CALCULATION ──
def compute_grid(top, pct, levels):
    bottom = top * (1 - pct/100)
    step = (top - bottom) / levels
    return bottom, step

# ── APP UI ──
st.set_page_config(page_title="XRP/BTC Grid Bot", layout="centered")
st.title("🟋 XRP/BTC Grid Bot (90 d Data)")

# 1. Live prices
btc_price, btc_change, xrp_price = fetch_live()

# 2. Historical & indicators
hist = fetch_history(HISTORY_DAYS)
row = hist.iloc[-1]
vol14 = row[f"vol{VOL_WINDOW}"]

# 3. Dynamic thresholds
mod_thresh = vol14
str_thresh = 2 * vol14

# 4. Confirmations
regime_ok = row["price"] > row["ema_trend"]
momentum_ok = row["sma_short"] > row["sma_long"]
rsi_ok = row["rsi"] < RSI_OVERBOUGHT

# 5. Trigger logic
if btc_change < mod_thresh:
    drop_pct = None
    status = f"No reset (BTC up {btc_change:.2f}% < {mod_thresh:.2f}%)"
elif btc_change <= str_thresh:
    drop_pct = mod_thresh
    status = f"🔔 Moderate reset: drop {mod_thresh:.2f}%"
else:
    drop_pct = str_thresh
    status = f"🔔 Strong reset: drop {str_thresh:.2f}%"

trigger = drop_pct is not None and regime_ok and momentum_ok and rsi_ok

# 6. Display live metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC/USD", f"${btc_price:,.2f}", f"{btc_change:.2f}%")
c2.metric("XRP/BTC", f"{xrp_price:.8f} BTC")
c3.metric("14 d Vol", f"{vol14:.2f}%")
c4.metric("Filters OK", f"{'✅' if (regime_ok and momentum_ok and rsi_ok) else '❌'}")

st.markdown(f"## {status}")
st.markdown(
    f"**Regime:** {'OK' if regime_ok else 'NO'}  |  "
    f"**Momentum:** {'OK' if momentum_ok else 'NO'}  |  "
    f"**RSI:** {'OK' if rsi_ok else 'NO'}"
)

# 7. Grid settings & calculation
levels = st.sidebar.number_input(
    "Grid levels", min_value=1, value=GRID_LEVELS_DEFAULT, step=1
)
if trigger:
    bottom, step = compute_grid(xrp_price, drop_pct, levels)
    st.write(f"**Grid top:** {xrp_price:.8f} BTC")
    st.write(f"**Grid bottom:** {bottom:.8f} BTC  (drop {drop_pct:.2f}%)")
    st.write(f"**Step size:** {step:.8f} BTC over {levels} levels")
else:
    st.write("No grid adjustment at this time.")
