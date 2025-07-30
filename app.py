import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# ── Auto-refresh every 60 s ──
st_autorefresh(interval=60_000, key="datarefresh")

# ── Streamlit Config ──
st.set_page_config(page_title="Scalping Grid Bot", layout="centered")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
st.markdown("_Free & Educational Use Only — Not Financial Advice_")
st.markdown("---")

# ── Sidebar Settings ──
st.sidebar.title("💰 Investment Settings")
inv_btc = st.sidebar.number_input("Total Investment (BTC)", min_value=1e-5, value=0.01, step=1e-5, format="%.5f")
min_order = st.sidebar.number_input("Min Order Size (BTC)", min_value=1e-6, value=5e-4, step=1e-6, format="%.6f")
RSI_OVERBOUGHT = st.sidebar.slider("RSI Overbought Threshold", min_value=60, max_value=90, value=75)
VOL_WINDOW = st.sidebar.slider("Volatility Window", min_value=7, max_value=30, value=14)
SMA_SHORT = st.sidebar.slider("Short-Term SMA", min_value=3, max_value=20, value=5)
SMA_LONG = st.sidebar.slider("Long-Term SMA", min_value=10, max_value=50, value=20)
EMA_TREND = st.sidebar.slider("Trend EMA", min_value=20, max_value=100, value=50)

# ── Constants ──
HISTORY_DAYS = 90
RSI_WINDOW = 14
GRID_MIN, GRID_MAX = 1, 30

# ── Fetch Historical Data ──
@st.cache_data(ttl=600)
def fetch_history(days):
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": days}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        prices = r.json()["prices"]
        df = pd.DataFrame(prices, columns=["ts", "price"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("date").resample("D").last().dropna()
        df["return"] = df["price"].pct_change() * 100
        df["vol"] = df["return"].rolling(VOL_WINDOW).std()
        df["sma_short"] = df["price"].rolling(SMA_SHORT).mean()
        df["sma_long"] = df["price"].rolling(SMA_LONG).mean()
        df["ema"] = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
        delta = df["price"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(RSI_WINDOW).mean()
        avg_loss = loss.rolling(RSI_WINDOW).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - 100 / (1 + rs)
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

# ── Fetch Live Prices ──
@st.cache_data(ttl=60)
def fetch_live():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "bitcoin,ripple",
            "vs_currencies": "usd,btc",
            "include_24hr_change": "true"
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        return {
            "BTC/USDT": (j["bitcoin"]["usd"], j["bitcoin"].get("usd_24h_change", 0)),
            "XRP/BTC": (j["ripple"]["btc"], j["ripple"].get("btc_24h_change", None))
        }
    except Exception as e:
        st.error(f"Error fetching live prices: {e}")
        return {}

# ── Grid Computation ──
def compute_grid(top, drop_pct, levels):
    bottom = top * (1 - drop_pct / 100)
    step = (top - bottom) / levels
    return bottom, step

# ── Load Data ──
hist = fetch_history(HISTORY_DAYS)
live = fetch_live()

if hist.empty or not live:
    st.stop()

latest = hist.iloc[-1]
vol14 = latest["vol"]

# ── Date & Context ──
now_london = datetime.now(pytz.timezone("Europe/London"))
st.markdown(f"**Date:** {now_london.strftime('%Y-%m-%d (%A) %H:%M %Z')}")
st.info("📉 _Historical data shows only closed daily candles (UTC)._")

# ── Price Chart ──
st.subheader("📈 BTC/USD Price & Indicators")
st.line_chart(hist[["price", "ema", "sma_short", "sma_long"]].rename(columns={
    "price": "Price", "ema": "EMA", "sma_short": "SMA Short", "sma_long": "SMA Long"
}))

# ── Backtest Conditions ──
mod_th = vol14
str_th = 2 * vol14
cond = (
    (hist["return"] >= mod_th) &
    (hist["price"] > hist["ema"]) &
    (hist["sma_short"] > hist["sma_long"]) &
    (hist["rsi"] < RSI_OVERBOUGHT)
)
trades = cond.sum()
wins = ((hist["price"].shift(-1) > hist["price"]) & cond).sum()
win_rate = wins / trades if trades else 0

# ── Grid Optimization ──
btc_change = live["BTC/USDT"][1]
drop_pct_btc = (
    mod_th if btc_change < mod_th else
    (str_th if btc_change > str_th else btc_change)
)
scores = [win_rate * (drop_pct_btc / L) for L in range(GRID_MIN, GRID_MAX + 1)]
opt_L = int(np.argmax(scores)) + GRID_MIN
few_L = max(GRID_MIN, opt_L - 10)
mor_L = min(GRID_MAX, opt_L + 10)

# ── Bot Execution ──
def run_bot(name, pair, price, pct_change):
    st.header(f"{name} ({pair})")
    st.write(f"- **Price:** {price:.8f}")
    if pct_change is not None:
        st.write(f"- **24h Change:** {pct_change:.2f}%")
    else:
        st.write("- **24h Change:** Not available")

    st.write(f"- **14d Volatility:** {vol14:.2f}%")
    filters_ok = (
        (latest["price"] > latest["ema"]) and
        (latest["sma_short"] > latest["sma_long"]) and
        (latest["rsi"] < RSI_OVERBOUGHT)
    )
    st.write(
        f"- **Filters:** Regime={'✅' if latest['price'] > latest['ema'] else '❌'}, "
        f"Momentum={'✅' if latest['sma_short'] > latest['sma_long'] else '❌'}, "
        f"RSI={'✅' if latest['rsi'] < RSI_OVERBOUGHT else '❌'}"
    )

    change = pct_change if pct_change is not None else hist["return"].iloc[-1]
    if change < mod_th:
        drop, status = None, f"No reset ({change:.2f}% < {mod_th:.2f}%)"
    elif change <= str_th:
        drop, status = mod_th, f"🔔 Moderate reset → drop {mod_th:.2f}%"
    else:
        drop, status = str_th, f"🔔 Strong reset → drop {str_th:.2f}%"
    st.markdown(f"**Status:** {status}")

    if drop is not None and filters_ok:
        st.subheader("📊 Grid Recommendations")

        L_choices = {
            "Most Profitable": opt_L,
            "Fewer": few_L,
            "More": mor_L
        }

        for label, L in L_choices.items():
            bottom, step = compute_grid(price, drop, L)
            per_order = inv_btc / L
            valid = per_order >= min_order
            st.markdown(
                f"**{label} ({L} levels)**  \n"
                f"- Lower: `{bottom:.8f}`  \n"
                f"- Upper: `{price:.8f}`   \n"
                f"- Step: `{step:.8f}`  \n"
                f"- Per‑Order: `{per_order:.6f}` BTC {'✅' if valid else '❌'}"
            )

        table = []
        for L in range(GRID_MIN, GRID_MAX + 1):
            _, step = compute_grid(price, drop, L)
            per = inv_btc / L
            table.append({
                "Levels": L,
                "Step (Δ)": f"{step:.8f}",
                "Per‑Order (BTC)": f"{per:.6f}",
                "Valid?": "✅" if per >= min_order else "❌"
            })
        st.write("### Grid Levels vs Per‑Order Size")
        st.table(pd.DataFrame(table))
    else:
        st.info("No grid recommendation at this time.")

    st.markdown("---")

# ── Run Bots ──
run_bot("XRP/BTC Bot", "XRP/BTC", *live["XRP/BTC"])
run_bot("BTC/USDT Bot", "BTC/USDT", *live["BTC/USDT"])

# ── Backtest Summary ──
st.subheader("⚙️ Strategy Backtest Summary (BTC/USD)")
st.write(f"- Total Signals: {trades}")
st.write(f"- Wins: {wins}")
st.write(f"- Win Rate: {win_rate * 100:.1f}%")
st.caption("_Note: Assumes 'win' if price rose the next day. No actual trade simulation used._")

# ── Signal Frequency ──
st.subheader("📊 Signal Frequency Estimates")
spd = trades / HISTORY_DAYS
st.write(f"- Per Day: {spd:.2f}")
st.write(f"- Per Week: {spd * 7:.2f}")
st.write(f"- Per Month: {spd * 30:.2f}")
st.write(f"- Per Year: {spd * 365:.2f}")
