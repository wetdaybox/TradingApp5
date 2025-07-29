import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 seconds
st_autorefresh(interval=60_000, key="datarefresh")

# --- Configuration ---
BACKTEST_DAYS = 180
VOL_WINDOWS = [7, 14, 30]
RSI_WINDOW = 14
SMA_SHORT = 5
SMA_LONG = 20
EMA_TREND = 200
RSI_OVERBOUGHT = 75
GRID_LEVELS = 10

BINANCE_API = "https://api.binance.com/api/v3"
COINGECKO_API = "https://api.coingecko.com/api/v3/simple/price"

# --- Fetch Live Data with Fallback ---
@st.cache_data(ttl=60)
def fetch_live():
    # Try Binance first
    try:
        r1 = requests.get(f"{BINANCE_API}/ticker/24hr", params={"symbol": "BTCUSDT"}, timeout=5)
        r1.raise_for_status()
        d1 = r1.json()
        btc_price = float(d1["lastPrice"])
        btc_change = float(d1["priceChangePercent"])
        r2 = requests.get(f"{BINANCE_API}/ticker/24hr", params={"symbol": "XRPBTC"}, timeout=5)
        r2.raise_for_status()
        d2 = r2.json()
        xrp_price = float(d2["lastPrice"])
        return btc_price, btc_change, xrp_price
    except Exception:
        # Fallback to CoinGecko
        params = {"ids": "bitcoin,ripple", "vs_currencies": "usd,btc", "include_24hr_change": "true"}
        r = requests.get(COINGECKO_API, params=params, timeout=10)
        r.raise_for_status()
        d = r.json()
        return (
            d["bitcoin"]["usd"],
            d["bitcoin"]["usd_24h_change"],
            d["ripple"]["btc"]
        )

# --- Fetch Historical Data ---
@st.cache_data(ttl=600)
def fetch_history(days):
    limit = days + EMA_TREND + 5
    r = requests.get(
        f"{BINANCE_API}/klines",
        params={"symbol": "BTCUSDT", "interval": "1d", "limit": limit},
        timeout=10
    )
    if r.status_code != 200:
        # fallback to CoinGecko full history
        cg = requests.get("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
                          params={"vs_currency": "usd", "days": days + EMA_TREND + 5}, timeout=10)
        cg.raise_for_status()
        prices = cg.json()["prices"]
        df = pd.DataFrame(prices, columns=["ts", "price"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("date").resample("D").last().dropna()
    else:
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "openTime","open","high","low","close","volume",
            "closeTime","quoteAssetVol","trades","takerBase","takerQuote","ignore"
        ])
        df["date"] = pd.to_datetime(df["openTime"], unit="ms")
        df = df.set_index("date")
        df["price"] = df["close"].astype(float)
    df["return"] = df["price"].pct_change() * 100
    for w in VOL_WINDOWS:
        df[f"vol{w}"] = df["return"].rolling(w).std()
    df["sma_short"] = df["price"].rolling(SMA_SHORT).mean()
    df["sma_long"] = df["price"].rolling(SMA_LONG).mean()
    df["ema_trend"] = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    delta = df["price"].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_WINDOW).mean()
    avg_loss = loss.rolling(RSI_WINDOW).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - 100 / (1 + rs)
    return df.dropna()

# --- Grid Computation ---
def compute_grid(top_price: float, drop_pct: float, levels: int):
    bottom = top_price * (1 - drop_pct / 100)
    step = (top_price - bottom) / levels
    return bottom, step

# --- Parameter Calibration ---
def calibrate(df):
    best = {"mult": None, "win": -1}
    base_vol = df["vol14"].iloc[-1]
    for m in np.linspace(0.5, 2.0, 16):
        thresh = m * base_vol
        wins = trades = 0
        for i in range(EMA_TREND, len(df) - 1):
            row = df.iloc[i]; nxt = df.iloc[i+1]
            cond = (
                (row["return"] >= thresh) and
                (row["price"] > row["ema_trend"]) and
                (row["sma_short"] > row["sma_long"]) and
                (row["rsi"] < RSI_OVERBOUGHT)
            )
            if cond:
                trades += 1
                if nxt["price"] > row["price"]:
                    wins += 1
        win_rate = wins / trades if trades else 0
        if win_rate > best["win"]:
            best = {"mult": m, "win": win_rate}
    return best

# --- Main App ---
st.set_page_config(page_title="Advanced XRP/BTC Grid Bot", layout="wide")
st.title("🟋 Advanced XRP/BTC Grid Bot")

# Live data
btc_price, btc_change, xrp_price = fetch_live()

# Historical and calibration
hist = fetch_history(BACKTEST_DAYS + EMA_TREND)
cal = calibrate(hist)
vol_threshold = cal["mult"] * hist["vol14"].iloc[-1]

# Indicator checks
row = hist.iloc[-1]
regime_ok = row["price"] > row["ema_trend"]
momentum_ok = row["sma_short"] > row["sma_long"]
rsi_ok = row["rsi"] < RSI_OVERBOUGHT

# Trigger
trigger = (btc_change >= vol_threshold) and regime_ok and momentum_ok and rsi_ok
drop_pct = vol_threshold if trigger else 0

# Display
c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC/USD", f"${btc_price:,.2f}", f"{btc_change:.2f}%")
c2.metric("XRP/BTC", f"{xrp_price:.8f} BTC")
c3.metric("Vol Threshold", f"{vol_threshold:.2f}%")
c4.metric("Win Rate", f"{cal['win']*100:.2f}%")

st.markdown(
    f"**Regime:** {'OK' if regime_ok else 'NOT OK'}  |  "
    f"**Momentum:** {'OK' if momentum_ok else 'NOT OK'}  |  "
    f"**RSI:** {'OK' if rsi_ok else 'NOT OK'}"
)
st.markdown(f"**Trigger:** {'YES' if trigger else 'NO'} (Threshold {vol_threshold:.2f}%)")

if trigger:
    bottom, step = compute_grid(xrp_price, drop_pct, GRID_LEVELS)
    st.write(
        f"**Grid Top:** {xrp_price:.8f} BTC  |  **Bottom:** {bottom:.8f} BTC  "
        f"(drop {drop_pct:.2f}%)"
    )
    st.write(f"**Step size:** {step:.8f} BTC over {GRID_LEVELS} levels")
else:
    st.write("No grid reset at this time.")

st.subheader("Calibration Result")
st.write(pd.DataFrame([cal]))
