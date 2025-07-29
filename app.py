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

# --- Fetch Live Data ---
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
        d['bitcoin']['usd'],
        d['bitcoin']['usd_24h_change'],
        d['ripple']['btc']
    )

# --- Fetch Historical Data ---
@st.cache_data(ttl=600)
def fetch_history(days):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    prices = r.json()['prices']
    df = pd.DataFrame(prices, columns=['ts', 'price'])
    df['date'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('date').resample('D').last().dropna()
    df['return'] = df['price'].pct_change() * 100
    # Volatility
    for w in VOL_WINDOWS:
        df[f'vol{w}'] = df['return'].rolling(w).std()
    # SMAs and EMA
    df['sma_short'] = df['price'].rolling(SMA_SHORT).mean()
    df['sma_long'] = df['price'].rolling(SMA_LONG).mean()
    df['ema_trend'] = df['price'].ewm(span=EMA_TREND, adjust=False).mean()
    # RSI
    delta = df['price'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_WINDOW).mean()
    avg_loss = loss.rolling(RSI_WINDOW).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - 100 / (1 + rs)
    return df.dropna()

# --- Grid Computation ---
def compute_grid(top_price: float, drop_pct: float, levels: int):
    bottom = top_price * (1 - drop_pct / 100)
    step = (top_price - bottom) / levels
    return bottom, step

# --- Parameter Calibration (Walk‑Forward) ---
def calibrate(df):
    best = {'mult': None, 'win': -1}
    base_vol = df['vol14'].iloc[-1]
    for m in np.linspace(0.5, 2.0, 16):
        thresh = m * base_vol
        wins = trades = 0
        for i in range(EMA_TREND, len(df) - 1):
            row = df.iloc[i]
            nxt = df.iloc[i + 1]
            c24 = row['return']
            cond = (
                (c24 >= thresh) and
                (row['price'] > row['ema_trend']) and
                (row['sma_short'] > row['sma_long']) and
                (row['rsi'] < RSI_OVERBOUGHT)
            )
            if cond:
                trades += 1
                if nxt['price'] > row['price']:
                    wins += 1
        win_rate = wins / trades if trades else 0
        if win_rate > best['win']:
            best = {'mult': m, 'win': win_rate}
    return best

# --- Main App ---
st.set_page_config(page_title="Advanced XRP/BTC Grid Bot", layout="wide")
st.title("🟋 Advanced XRP/BTC Grid Bot")

# Live data
btc_price, btc_change, xrp_price = fetch_live()

# Historical & Calibration
hist = fetch_history(BACKTEST_DAYS + EMA_TREND)
cal = calibrate(hist)
vol_threshold = cal['mult'] * hist['vol14'].iloc[-1]

# Indicator checks
regime_ok = btc_price > hist['ema_trend'].iloc[-1]
momentum_ok = hist['sma_short'].iloc[-1] > hist['sma_long'].iloc[-1]
rsi_ok = hist['rsi'].iloc[-1] < RSI_OVERBOUGHT

# Determine trigger
trigger = (
    (btc_change >= vol_threshold) and
    regime_ok and
    momentum_ok and
    rsi_ok
)
drop_pct = vol_threshold if trigger else 0

# Display live metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("BTC/USD", f"${btc_price:,.2f}", f"{btc_change:.2f}%")
col2.metric("XRP/BTC", f"{xrp_price:.8f} BTC")
col3.metric("Vol Threshold", f"{vol_threshold:.2f}%")
col4.metric("Backtest Win%", f"{cal['win']*100:.2f}%")

# Status line
st.markdown(
    f"**Regime:** {'OK' if regime_ok else 'NOT OK'}  |  "
    f"**Momentum:** {'OK' if momentum_ok else 'NOT OK'}  |  "
    f"**RSI:** {'OK' if rsi_ok else 'NOT OK'}"
)
st.markdown(f"**Trigger:** {'YES' if trigger else 'NO'} (Threshold {vol_threshold:.2f}%)")

# Grid display
if trigger:
    bottom, step = compute_grid(xrp_price, drop_pct, GRID_LEVELS)
    st.write(
        f"**Grid Top:** {xrp_price:.8f} BTC  |  **Bottom:** {bottom:.8f} BTC  "
        f"(drop {drop_pct:.2f}%)"
    )
    st.write(f"**Step size:** {step:.8f} BTC per level ({GRID_LEVELS} levels)")
else:
    st.write("No grid reset at this time.")

# Backtest summary
st.subheader("Backtest Calibration")
st.write(pd.DataFrame({
    'Multiplier': [cal['mult']],
    'Win Rate': [cal['win']]
}))
