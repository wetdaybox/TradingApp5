import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ── Auto‑refresh every 60 s ──
st_autorefresh(interval=60_000, key="datarefresh")

# ── Configuration ──
HISTORY_DAYS   = 90   # <=90 so CoinGecko market_chart works reliably
VOL_WINDOW     = 14
RSI_WINDOW     = 14
SMA_SHORT      = 5
SMA_LONG       = 20
EMA_TREND      = 50
RSI_OVERBOUGHT = 75
GRID_LEVELS    = 10

# ── Fetch live BTC/USD & XRP/BTC from CoinGecko ──
@st.cache_data(ttl=60)
def fetch_live():
    r = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={
            "ids": "bitcoin,ripple",
            "vs_currencies": "usd,btc",
            "include_24hr_change": "true"
        },
        timeout=10
    )
    r.raise_for_status()
    j = r.json()
    return (
        j["bitcoin"]["usd"],
        j["bitcoin"]["usd_24h_change"],
        j["ripple"]["btc"]
    )

# ── Fetch 90 d history & compute indicators ──
@st.cache_data(ttl=600)
def fetch_history(days):
    r = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
        params={"vs_currency": "usd", "days": days},
        timeout=10
    )
    r.raise_for_status()
    prices = r.json()["prices"]
    df = pd.DataFrame(prices, columns=["ts", "price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change() * 100
    # Volatility
    df["vol14"] = df["return"].rolling(VOL_WINDOW).std()
    # SMA & EMA
    df["sma5"]  = df["price"].rolling(SMA_SHORT).mean()
    df["sma20"] = df["price"].rolling(SMA_LONG).mean()
    df["ema50"] = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    # RSI
    delta = df["price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_WINDOW).mean()
    avg_loss = loss.rolling(RSI_WINDOW).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - 100 / (1 + rs)
    return df.dropna()

# ── Grid computation ──
def compute_grid(top, pct, levels):
    bottom = top * (1 - pct/100)
    step   = (top - bottom) / levels
    return bottom, step

# ── App UI ──
st.set_page_config(layout="centered")
st.title("🟋 XRP/BTC Grid Bot with Filters & Backtest")

# 1️⃣ Live data
btc_price, btc_change, xrp_price = fetch_live()

# 2️⃣ History + indicators
hist = fetch_history(HISTORY_DAYS)
row  = hist.iloc[-1]
vol14 = row["vol14"]

# 3️⃣ Dynamic thresholds
mod_thresh = vol14
str_thresh = 2 * vol14

# 4️⃣ Filter checks
regime_ok   = row["price"] > row["ema50"]
momentum_ok = row["sma5"]   > row["sma20"]
rsi_ok      = row["rsi"]    < RSI_OVERBOUGHT

# 5️⃣ Trigger logic
if btc_change < mod_thresh:
    drop_pct = None
    status   = f"No reset (BTC up {btc_change:.2f}% < {mod_thresh:.2f}%)"
elif btc_change <= str_thresh:
    drop_pct = mod_thresh
    status   = f"🔔 Moderate reset → drop {mod_thresh:.2f}%"
else:
    drop_pct = str_thresh
    status   = f"🔔 Strong reset → drop {str_thresh:.2f}%"

trigger = drop_pct is not None and regime_ok and momentum_ok and rsi_ok

# 6️⃣ Display live metrics & filters
c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC/USD", f"${btc_price:,.2f}", f"{btc_change:.2f}%")
c2.metric("XRP/BTC", f"{xrp_price:.8f} BTC")
c3.metric("14 d Vol", f"{vol14:.2f}%")
c4.metric("Filters OK", f"{'✅' if (regime_ok and momentum_ok and rsi_ok) else '❌'}")

st.markdown(f"## {status}")
st.markdown(
    f"**Regime (EMA50):** {'OK' if regime_ok else 'NO'}  |  "
    f"**Momentum (5/20 SMA):** {'OK' if momentum_ok else 'NO'}  |  "
    f"**RSI<75:** {'OK' if rsi_ok else 'NO'}"
)

# 7️⃣ Grid settings & calculation
levels = st.sidebar.number_input("Grid levels", 1, 50, GRID_LEVELS)
if trigger:
    bottom, step = compute_grid(xrp_price, drop_pct, levels)
    st.write(f"**Grid Top:** {xrp_price:.8f} BTC   |   **Bottom:** {bottom:.8f} BTC")
    st.write(f"**Step size:** {step:.8f} BTC over {levels} levels")
else:
    st.write("No grid adjustment at this time.")

# 8️⃣ 90 d Backtest
trades = wins = 0
for i in range(len(hist)-1):
    r0 = hist["return"].iloc[i]
    f0 = (
        r0>=mod_thresh and
        hist["price"].iloc[i]>hist["ema50"].iloc[i] and
        hist["sma5"].iloc[i]>hist["sma20"].iloc[i] and
        hist["rsi"].iloc[i]<RSI_OVERBOUGHT
    )
    if f0:
        trades += 1
        if hist["price"].iloc[i+1] > hist["price"].iloc[i]:
            wins += 1
win_rate = wins/trades*100 if trades else 0
st.subheader("Backtest (90 d)")
st.write(f"Signals: {trades}, Wins: {wins}, Win Rate: {win_rate:.1f}%")
