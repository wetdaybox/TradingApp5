import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=60_000, key="refresh")
st.set_page_config(page_title="Grid Bot Reset Assistant", layout="centered")

# ── Sidebar Inputs ──
st.sidebar.title("💰 Investment Settings")
inv_btc = st.sidebar.number_input("Investment (BTC)", min_value=0.001, value=0.01, step=0.001)
min_order = st.sidebar.number_input("Min Order Size (BTC)", min_value=0.0001, value=0.0005, step=0.0001)

# ── Config ──
HISTORY_DAYS = 90
VOL_WINDOW = 14
SMA_SHORT, SMA_LONG = 5, 20
EMA_TREND = 50
RSI_WINDOW = 14
RSI_OVERBOUGHT = 75
GRID_MIN, GRID_MAX = 1, 30

# ── Helper Functions ──
def fetch_coingecko_price(ids, vs):
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": ids, "vs_currencies": vs, "include_24hr_change": "true"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def fetch_btc_history():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    r = requests.get(url, params={"vs_currency": "usd", "days": HISTORY_DAYS}, timeout=10)
    df = pd.DataFrame(r.json()["prices"], columns=["ts", "price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change() * 100
    df["vol"] = df["return"].rolling(VOL_WINDOW).std()
    df["sma5"] = df["price"].rolling(SMA_SHORT).mean()
    df["sma20"] = df["price"].rolling(SMA_LONG).mean()
    df["ema"] = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    delta = df["price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(RSI_WINDOW).mean() / loss.rolling(RSI_WINDOW).mean().replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)
    return df.dropna()

@st.cache_data(ttl=600)
def simulate_xrpbtc_history():
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 0.0015, size=HISTORY_DAYS)) + 0.000015
    prices = 0.000015 + np.abs(prices)
    df = pd.DataFrame({
        "date": pd.date_range(end=pd.Timestamp.today(), periods=HISTORY_DAYS),
        "price": prices
    })
    df["return"] = df["price"].pct_change() * 100
    df["vol10"] = df["return"].rolling(10).std()
    df["mean10"] = df["price"].rolling(10).mean()
    df["vol_rising"] = df["vol10"] > df["vol10"].shift(1)
    df["mean_revert"] = (df["price"] < df["mean10"]) & df["vol_rising"]
    return df.dropna()

def compute_grid(top, drop_pct, levels):
    bottom = top * (1 - drop_pct / 100)
    step = (top - bottom) / levels
    return bottom, step

def display_btc_bot(df, price, pct_change):
    latest = df.iloc[-1]
    mod_th = latest["vol"]
    str_th = 2 * mod_th
    cond = (
        (df["return"] >= mod_th) &
        (df["price"] > df["ema"]) &
        (df["sma5"] > df["sma20"]) &
        (df["rsi"] < RSI_OVERBOUGHT)
    )
    trades = cond.sum()
    wins = ((df["price"].shift(-1) > df["price"]) & cond).sum()
    win_rate = wins / trades if trades else 0

    if pct_change < mod_th:
        drop, status = None, f"No reset (Δ {pct_change:.2f}% < {mod_th:.2f}%)"
    elif pct_change <= str_th:
        drop, status = mod_th, f"🔔 Moderate reset: drop {mod_th:.2f}%"
    else:
        drop, status = str_th, f"🔔 Strong reset: drop {str_th:.2f}%"

    filters_ok = (
        (latest["price"] > latest["ema"]) and
        (latest["sma5"] > latest["sma20"]) and
        (latest["rsi"] < RSI_OVERBOUGHT)
    )

    st.markdown(f"**Current Price:** ${price:.2f}  \n**24h Change:** {pct_change:.2f}%")
    st.markdown(f"**Volatility (14d):** {mod_th:.2f}%")
    st.markdown(f"**Signal Status:** {status}")
    st.markdown(f"**Filters Passed:** {'✅' if filters_ok else '❌'}")

    if drop and filters_ok:
        scores = [win_rate * (drop / L) for L in range(GRID_MIN, GRID_MAX + 1)]
        opt_L = int(np.argmax(scores)) + GRID_MIN
        bottom, step = compute_grid(price, drop, opt_L)
        per_order = inv_btc / opt_L
        valid = per_order >= min_order
        st.subheader("📐 Recommended Grid")
        st.markdown(f"""
- **Upper Bound:** `{price:.2f}`  
- **Lower Bound:** `{bottom:.2f}`  
- **Levels:** `{opt_L}`  
- **Step Size:** `{step:.4f}`  
- **Per Order:** `{per_order:.6f}` BTC {'✅' if valid else '❌'}
        """)
    else:
        st.info("No grid recommendation right now.")

    with st.expander("📊 Backtest Summary"):
        st.write(f"- Signals: {trades}")
        st.write(f"- Wins: {wins}")
        st.write(f"- Win Rate: {win_rate*100:.1f}%")

def display_xrp_bot(df, price):
    latest = df.iloc[-1]
    signal = latest["mean_revert"]
    triggered = "✅ Reset suggested (mean reversion)" if signal else "❌ No reset"
    upper = latest["mean10"]
    drop = latest["vol10"]
    lower = price - (price * drop / 100)
    step = (upper - lower) / 10
    per_order = inv_btc / 10
    valid = per_order >= min_order

    st.markdown(f"**Current Price:** {price:.6f} BTC")
    st.markdown(f"**10-day Mean:** {upper:.6f} BTC")
    st.markdown(f"**10-day Volatility:** {drop:.2f}%")
    st.markdown(f"**Signal:** {triggered}")

    if signal:
        st.subheader("📐 Recommended Grid")
        st.markdown(f"""
- **Upper Bound:** `{upper:.6f}`  
- **Lower Bound:** `{lower:.6f}`  
- **Levels:** `10`  
- **Step Size:** `{step:.8f}`  
- **Per Order:** `{per_order:.6f}` BTC {'✅' if valid else '❌'}
        """)
    else:
        st.info("No grid recommendation at this time.")

    with st.expander("📊 Signal History"):
        total = df["mean_revert"].sum()
        st.write(f"- Signals in last 90 days: {total}")
        st.write(f"- Current trigger: {'✅' if signal else '❌'}")

# ── Fetch Prices and Data ──
btc_hist = fetch_btc_history()
xrp_hist = simulate_xrpbtc_history()

prices = fetch_coingecko_price("bitcoin,ripple", "usd,btc")
btc_price = prices["bitcoin"]["usd"]
btc_change = prices["bitcoin"].get("usd_24h_change", 0)
xrp_btc_price = prices["ripple"]["btc"]

now = datetime.now(pytz.timezone("Europe/London"))
st.caption(f"📅 Last updated: {now.strftime('%Y-%m-%d %H:%M %Z')}")

# ── Tabs Layout ──
tab1, tab2 = st.tabs(["🟡 BTC/USDT Bot", "🟣 XRP/BTC Bot"])

with tab1:
    st.header("BTC/USDT Reset Assistant")
    display_btc_bot(btc_hist, btc_price, btc_change)

with tab2:
    st.header("XRP/BTC Reset Assistant")
    display_xrp_bot(xrp_hist, xrp_btc_price)

# ── Expander Info ──
with st.expander("ℹ️ About This Tool"):
    st.markdown("""
This app provides **manual grid trading signals** for use with the **Crypto.com Exchange Grid Bot**.
- It does **not place trades**
- It calculates ideal **grid levels**, **ranges**, and **reset timing** for:
    - 🟡 **BTC/USDT** (trend/momentum/volatility)
    - 🟣 **XRP/BTC** (mean reversion)

Update your live bot parameters manually when a signal is triggered ✅
""")
