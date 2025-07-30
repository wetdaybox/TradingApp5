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
st.set_page_config(page_title="Grid Reset Assistant", layout="centered")
st.title("🔁 Crypto.com Grid Bot Reset Assistant")

# ── Optional Info Hidden by Default ──
with st.expander("ℹ️ How This Works"):
    st.markdown("""
    This tool **does not place trades**. It is designed to be used **manually** alongside the **Crypto.com Grid Trading Bot**.
    
    - 📉 It detects when to reset your active grid based on trend, momentum, and volatility.
    - 📐 It calculates optimal **upper/lower bounds**, **grid step size**, and **number of levels**.
    - 💡 All decisions are based on current BTC/USD market conditions and past volatility.
    
    🔗 Use this data to manually update your grid in the **Crypto.com exchange bot** interface.
    """)

# ── Sidebar: User Configurations ──
st.sidebar.title("💰 Investment Settings")
inv_btc = st.sidebar.number_input("Total Investment (BTC)", min_value=1e-5, value=0.01, step=1e-5, format="%.5f")
min_order = st.sidebar.number_input("Min Order Size (BTC)", min_value=1e-6, value=5e-4, step=1e-6, format="%.6f")
RSI_OVERBOUGHT = st.sidebar.slider("RSI Overbought Threshold", 60, 90, 75)
VOL_WINDOW = st.sidebar.slider("Volatility Window", 7, 30, 14)
SMA_SHORT = st.sidebar.slider("Short SMA (Momentum)", 3, 20, 5)
SMA_LONG = st.sidebar.slider("Long SMA (Trend)", 10, 50, 20)
EMA_TREND = st.sidebar.slider("EMA (Trend Filter)", 20, 100, 50)

# ── Constants ──
HISTORY_DAYS = 90
RSI_WINDOW = 14
GRID_MIN, GRID_MAX = 1, 30

# ── Load Data ──
@st.cache_data(ttl=600)
def fetch_history(days):
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": days}
        r = requests.get(url, params=params, timeout=10); r.raise_for_status()
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
        st.error(f"Failed to fetch historical data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_live():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "bitcoin",
            "vs_currencies": "usd",
            "include_24hr_change": "true"
        }
        r = requests.get(url, params=params, timeout=10); r.raise_for_status()
        j = r.json()
        return j["bitcoin"]["usd"], j["bitcoin"].get("usd_24h_change", 0)
    except Exception as e:
        st.error(f"Failed to fetch live price: {e}")
        return None, None

def compute_grid(top, drop_pct, levels):
    bottom = top * (1 - drop_pct / 100)
    step = (top - bottom) / levels
    return bottom, step

# ── Data Load ──
hist = fetch_history(HISTORY_DAYS)
price, pct_change = fetch_live()

if hist.empty or price is None:
    st.warning("Unable to load data. Please wait or try again later.")
    st.stop()

latest = hist.iloc[-1]
volatility = latest["vol"]

# ── Display Date & Info ──
now = datetime.now(pytz.timezone("Europe/London"))
st.markdown(f"**Current Time (London):** {now.strftime('%Y-%m-%d %H:%M %Z')}")

# ── Backtest Filters ──
mod_th = volatility
str_th = 2 * volatility
conditions = (
    (hist["return"] >= mod_th) &
    (hist["price"] > hist["ema"]) &
    (hist["sma_short"] > hist["sma_long"]) &
    (hist["rsi"] < RSI_OVERBOUGHT)
)
signals = conditions.sum()
wins = ((hist["price"].shift(-1) > hist["price"]) & conditions).sum()
win_rate = wins / signals if signals else 0

# ── Determine Drop Level ──
if pct_change < mod_th:
    drop, signal_type = None, "No reset: volatility too low"
elif pct_change <= str_th:
    drop, signal_type = mod_th, f"🔔 Moderate reset suggested ({mod_th:.2f}% drop)"
else:
    drop, signal_type = str_th, f"🔔 Strong reset suggested ({str_th:.2f}% drop)"

filters_pass = (
    (latest["price"] > latest["ema"]) and
    (latest["sma_short"] > latest["sma_long"]) and
    (latest["rsi"] < RSI_OVERBOUGHT)
)

st.header("📊 Grid Reset Signal")
st.markdown(f"- **Price:** ${price:.2f}")
st.markdown(f"- **24h Change:** {pct_change:.2f}%")
st.markdown(f"- **14d Volatility:** {volatility:.2f}%")
st.markdown(f"- **Signal Status:** {signal_type}")
st.markdown(f"- **Filters Passed:** {'✅' if filters_pass else '❌'}")

if filters_pass and drop:
    st.subheader("📐 Recommended Grid Configurations")
    scores = [win_rate * (drop / L) for L in range(GRID_MIN, GRID_MAX + 1)]
    opt_L = int(np.argmax(scores)) + GRID_MIN
    few_L = max(GRID_MIN, opt_L - 10)
    mor_L = min(GRID_MAX, opt_L + 10)

    L_choices = {
        "Most Profitable": opt_L,
        "Fewer Levels": few_L,
        "More Levels": mor_L
    }

    for label, L in L_choices.items():
        bottom, step = compute_grid(price, drop, L)
        per_order = inv_btc / L
        valid = per_order >= min_order
        st.markdown(f"""
        **{label} ({L} levels)**  
        - Upper: `{price:.2f}`  
        - Lower: `{bottom:.2f}`  
        - Step: `{step:.4f}`  
        - Per Order: `{per_order:.6f}` BTC {'✅' if valid else '❌'}
        """)

    # ── Copyable Summary ──
    st.subheader("📋 Copyable Summary")
    summary = f"""
Grid Reset Signal ✅  
Upper Bound: {price:.2f}  
Drop %: {drop:.2f}  
Best Grid Levels: {opt_L}  
Per Order Size: {inv_btc/opt_L:.6f} BTC  
Minimum Order Met: {"✅" if inv_btc/opt_L >= min_order else "❌"}  
    """.strip()
    st.code(summary, language="markdown")

    # ── Grid Table ──
    with st.expander("📑 Full Level Table (1–30 Levels)"):
        table = []
        for L in range(GRID_MIN, GRID_MAX + 1):
            _, step = compute_grid(price, drop, L)
            per = inv_btc / L
            table.append({
                "Levels": L,
                "Step (Δ)": f"{step:.4f}",
                "Per Order (BTC)": f"{per:.6f}",
                "Valid?": "✅" if per >= min_order else "❌"
            })
        st.table(pd.DataFrame(table))
else:
    st.info("🕒 No grid recommendation at this time based on filters or volatility.")

# ── Signal Frequency Summary ──
with st.expander("📈 Backtest Signal Summary (Past 90 Days)"):
    st.write(f"- Signals: {signals}")
    st.write(f"- Wins: {wins}")
    st.write(f"- Win Rate: {win_rate*100:.1f}%")
    daily = signals / HISTORY_DAYS
    st.write(f"- Estimated: {daily:.2f}/day, {daily*7:.1f}/week, {daily*30:.1f}/month")

# ── Footer ──
st.caption("📎 Use this tool with Crypto.com’s Grid Bot by updating grid levels manually when signaled.")
