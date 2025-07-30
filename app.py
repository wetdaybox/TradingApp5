import streamlit as st
import requests
import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh

# ── Auto‑refresh every 60 s ──
st_autorefresh(interval=60_000, key="datarefresh")

# ── Configuration ──
HISTORY_DAYS   = 90
VOL_WINDOW     = 14
RSI_WINDOW     = 14
SMA_SHORT      = 5
SMA_LONG       = 20
EMA_TREND      = 50
RSI_OVERBOUGHT = 75

# “Optimal” and alt grid levels
GRID_PRIMARY   = 20  # most profitable
GRID_FEWER     = 10  # fewer levels
GRID_MORE      = 30  # more levels

# ── Fetch live data ──
@st.cache_data(ttl=60)
def fetch_live():
    r = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={
            "ids": "bitcoin,ripple",
            "vs_currencies": "usd,btc",
            "include_24hr_change": "true"
        }, timeout=10
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
        params={"vs_currency": "usd", "days": days}, timeout=10
    )
    r.raise_for_status()
    prices = r.json()["prices"]
    df = pd.DataFrame(prices, columns=["ts", "price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change() * 100
    df["vol14"] = df["return"].rolling(VOL_WINDOW).std()
    df["sma5"] = df["price"].rolling(SMA_SHORT).mean()
    df["sma20"] = df["price"].rolling(SMA_LONG).mean()
    df["ema50"] = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    delta = df["price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_WINDOW).mean()
    avg_loss = loss.rolling(RSI_WINDOW).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - 100 / (1 + rs)
    return df.dropna()

# ── Grid calc ──
def compute_grid(top, pct, levels):
    bottom = top * (1 - pct/100)
    step   = (top - bottom) / levels
    return bottom, step

# ── UI ──
st.set_page_config(layout="centered")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")

# 1) Live data
btc_price, btc_change, xrp_price = fetch_live()

# 2) History + indicators
hist = fetch_history(HISTORY_DAYS)
row  = hist.iloc[-1]
vol14 = row["vol14"]

# 3) Thresholds & filters
mod_thresh   = vol14
str_thresh   = 2 * vol14
regime_ok    = row["price"] > row["ema50"]
momentum_ok  = row["sma5"]   > row["sma20"]
rsi_ok       = row["rsi"]    < RSI_OVERBOUGHT

# 4) Trigger logic
if btc_change < mod_thresh:
    drop_pct, status = None, f"No reset (BTC +{btc_change:.2f}% < {mod_thresh:.2f}%)"
elif btc_change <= str_thresh:
    drop_pct, status = mod_thresh, f"🔔 Moderate reset → drop {mod_thresh:.2f}%"
else:
    drop_pct, status = str_thresh, f"🔔 Strong reset → drop {str_thresh:.2f}%"
trigger = drop_pct is not None and regime_ok and momentum_ok and rsi_ok

# 5) Display metrics & filters
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"**BTC / USD**\n${btc_price:,.2f} ({btc_change:.2f}%)")
c2.markdown(f"**XRP / BTC**\n{xrp_price:.8f} BTC")
c3.markdown(f"**14 d Vol**\n{vol14:.2f}%")
c4.markdown(f"**Filters OK**\n{'✅' if (regime_ok and momentum_ok and rsi_ok) else '❌'}")

st.markdown(f"## {status}")
st.markdown(
    f"**Regime (EMA50):** {'OK' if regime_ok else 'NO'}  |  "
    f"**Momentum (5/20 SMA):** {'OK' if momentum_ok else 'NO'}  |  "
    f"**RSI<75:** {'OK' if rsi_ok else 'NO'}"
)

# 6) Grid instructions
if trigger:
    # Primary
    bottom_p, step_p = compute_grid(xrp_price, drop_pct, GRID_PRIMARY)
    st.write("### 🚀 Recommended Setup (Most Profitable)")
    st.markdown(f"""
- **Mode:** Geometric  
- **Lower Price:** `{bottom_p:.8f} BTC`  
- **Upper Price:** `{xrp_price:.8f} BTC`  
- **Grid Number:** `{GRID_PRIMARY}`  
- **Trailing Up:** On  
- **Stop‑Loss:** `{(bottom_p*0.95):.8f} BTC` (5% below)  
- **Take‑Profit:** `{(xrp_price*1.05):.8f} BTC` (5% above)  
""")

    # Alternatives
    st.write("### ⚖️ Alternative Grid Levels")
    for lvl, label in [(GRID_FEWER, "Fewer Levels (Coarser)"), (GRID_MORE, "More Levels (Finer)")]:
        bottom_a, step_a = compute_grid(xrp_price, drop_pct, lvl)
        st.markdown(f"""
**{label}**  
- Grid Number: `{lvl}`  
- Step Size: `{step_a:.8f} BTC`  
  - {'Larger steps capture bigger swings but fewer trades.' if lvl==GRID_FEWER else 'Smaller steps give more granularity but may incur more fees.'}
""")
else:
    st.write("No grid adjustment at this time.")

# 7) 90 d Backtest
trades = wins = 0
for i in range(len(hist)-1):
    cond = (
        hist["return"].iloc[i] >= mod_thresh and
        hist["price"].iloc[i] > hist["ema50"].iloc[i] and
        hist["sma5"].iloc[i] > hist["sma20"].iloc[i] and
        hist["rsi"].iloc[i] < RSI_OVERBOUGHT
    )
    if cond:
        trades += 1
        if hist["price"].iloc[i+1] > hist["price"].iloc[i]:
            wins += 1
win_rate = wins/trades*100 if trades else 0
st.subheader("Backtest (90 d)")
st.write(f"Signals: {trades}, Wins: {wins}, Win Rate: {win_rate:.1f}%")
