import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# ── Auto-refresh every 60 s ──
st_autorefresh(interval=60_000, key="datarefresh")

# ── Configuration ──
HISTORY_DAYS   = 90
VOL_WINDOW     = 14
RSI_WINDOW     = 14
SMA_SHORT      = 5
SMA_LONG       = 20
EMA_TREND      = 50
RSI_OVERBOUGHT = 75

# Grid‑level presets for XRP/BTC
GRID_PRIMARY = 20
GRID_FEWER   = 10
GRID_MORE    = 30

# ── Fetch 90 d BTC/USD history & compute indicators ──
@st.cache_data(ttl=600)
def fetch_history(days):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days}
    r = requests.get(url, params=params, timeout=10); r.raise_for_status()
    prices = r.json()["prices"]
    df = pd.DataFrame(prices, columns=["ts", "price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change() * 100
    df["vol14"]  = df["return"].rolling(VOL_WINDOW).std()
    df["sma5"]   = df["price"].rolling(SMA_SHORT).mean()
    df["sma20"]  = df["price"].rolling(SMA_LONG).mean()
    df["ema50"]  = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    delta       = df["price"].diff()
    gain        = delta.clip(lower=0)
    loss        = -delta.clip(upper=0)
    avg_gain    = gain.rolling(RSI_WINDOW).mean()
    avg_loss    = loss.rolling(RSI_WINDOW).mean()
    df["rsi"]   = 100 - 100 / (1 + avg_gain / avg_loss)
    return df.dropna()

# ── Fetch live prices for both pairs ──
@st.cache_data(ttl=60)
def fetch_live():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin,ripple",
        "vs_currencies": "usd,btc",
        "include_24hr_change": "true"
    }
    r = requests.get(url, params=params, timeout=10); r.raise_for_status()
    j = r.json()
    return {
        "BTC/USDT": (j["bitcoin"]["usd"], j["bitcoin"]["usd_24h_change"]),
        "XRP/BTC": (j["ripple"]["btc"], None)
    }

# ── Compute grid bounds ──
def compute_grid(top, drop_pct, levels):
    bottom = top * (1 - drop_pct / 100)
    step   = (top - bottom) / levels
    return bottom, step

# ── Sidebar inputs ──
st.sidebar.title("💰 Investment Settings")
inv_btc   = st.sidebar.number_input("Total Investment (BTC)", min_value=1e-5, value=0.01, step=1e-5, format="%.5f")
min_order = st.sidebar.number_input("Min Order Size (BTC)",   min_value=1e-6, value=5e-4,  step=1e-6, format="%.6f")

# ── Load data ──
hist   = fetch_history(HISTORY_DAYS)
live   = fetch_live()
latest = hist.iloc[-1]
vol14  = latest["vol14"]

# ── Page setup ──
st.set_page_config(layout="centered")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")

# ── Display current date in London time ──
now_london = datetime.now(pytz.timezone("Europe/London"))
st.markdown(f"**Date:** {now_london.strftime('%Y-%m-%d (%A) %H:%M %Z')}")

# ── Brief note on history lag ──
st.info(
    "🔍 _Note: Historical data shows only fully closed daily candles (UTC). "
    "It may lag by up to one day until the new candle completes._"
)

# ── Shared backtest of trigger on BTC/USD ──
mod_th = vol14
str_th = 2 * vol14
cond   = (
    (hist["return"] >= mod_th) &
    (hist["price"] > hist["ema50"]) &
    (hist["sma5"] > hist["sma20"]) &
    (hist["rsi"] < RSI_OVERBOUGHT)
)
trades   = int(cond.sum())
wins     = int(((hist["price"].shift(-1) > hist["price"]) & cond).sum())
win_rate = wins / trades if trades else 0

# ── Optimize grid count for BTC/USDT ──
btc_change    = live["BTC/USDT"][1]
drop_pct_btc  = mod_th if btc_change < mod_th else (str_th if btc_change > str_th else btc_change)
scores        = [win_rate * (drop_pct_btc / L) for L in range(1, 31)]
opt_L         = int(np.argmax(scores)) + 1
few_L         = max(1, opt_L - 10)
mor_L         = min(30, opt_L + 10)

# ── Bot runner function ──
def run_bot(name, pair, price, pct_change):
    st.header(f"{name} ({pair})")
    st.write(f"- **Price:** {price:.8f}")
    if pct_change is not None:
        st.write(f"- **24 h Change:** {pct_change:.2f}%")
    st.write(f"- **14 d Volatility:** {vol14:.2f}%")
    filters_ok = (
        (latest["price"] > latest["ema50"]) and
        (latest["sma5"] > latest["sma20"]) and
        (latest["rsi"] < RSI_OVERBOUGHT)
    )
    st.write(
        f"- **Filters:** Regime={'✅' if latest['price']>latest['ema50'] else '❌'}, "
        f"Momentum={'✅' if latest['sma5']>latest['sma20'] else '❌'}, "
        f"RSI={'✅' if latest['rsi']<RSI_OVERBOUGHT else '❌'}"
    )

    change = pct_change if pct_change is not None else hist["return"].iloc[-1]
    if change < mod_th:
        drop, status = None, f"No reset ({change:.2f}% < {mod_th:.2f}%)"
    elif change <= str_th:
        drop, status = mod_th,  f"🔔 Moderate reset → drop {mod_th:.2f}%"
    else:
        drop, status = str_th,  f"🔔 Strong reset → drop {str_th:.2f}%"
    st.markdown(f"**Status:** {status}")

    if drop is not None and filters_ok:
        st.subheader("📈 Grid Recommendations")

        # Select grid levels per pair
        if pair == "BTC/USDT":
            primary, fewer, more = opt_L, few_L, mor_L
        else:
            primary, fewer, more = GRID_PRIMARY, GRID_FEWER, GRID_MORE

        for L, label in [(primary, "Most Profitable"), (fewer, "Fewer"), (more, "More")]:
            bottom, step = compute_grid(price, drop, L)
            per = inv_btc / L
            valid = per >= min_order
            st.markdown(
                f"**{label} ({L} levels)**  \n"
                f"- Lower: `{bottom:.8f}`  \n"
                f"- Upper: `{price:.8f}`   \n"
                f"- Step: `{step:.8f}`  \n"
                f"- Per‑Order: `{per:.6f}` BTC {'✅' if valid else '❌'}"
            )

        st.write("### Grid Levels 1–30 vs. Per‑Order Size")
        table = []
        for L in range(1, 31):
            _, step = compute_grid(price, drop, L)
            per = inv_btc / L
            table.append({
                "Levels": L,
                "Step (ΔBTC)": f"{step:.8f}",
                "Per‑Order (BTC)": f"{per:.6f}",
                "Valid?": "✅" if per >= min_order else "❌"
            })
        st.table(pd.DataFrame(table))
    else:
        st.info("No grid adjustment at this time.")

    st.markdown("---")

# ── Run both bots ──
run_bot("XRP/BTC Bot",  "XRP/BTC",  *live["XRP/BTC"])
run_bot("BTC/USDT Bot", "BTC/USDT", *live["BTC/USDT"])

# ── Backtest summary ──
st.subheader("⚙️ Strategy Backtest (BTC/USD Signals over 90 d)")
st.write(f"- Signals: {trades} | Wins: {wins} | Win Rate: {win_rate*100:.1f}%")

# ── Expected signal frequency ──
signals_per_day   = trades / HISTORY_DAYS
signals_per_week  = signals_per_day * 7
signals_per_month = signals_per_day * 30
signals_per_year  = signals_per_day * 365

st.subheader("📅 Expected Signal Frequency")
st.write(f"- **Per day:**   {signals_per_day:.2f} signals")
st.write(f"- **Per week:**  {signals_per_week:.2f} signals")
st.write(f"- **Per month:** {signals_per_month:.2f} signals")
st.write(f"- **Per year:**  {signals_per_year:.2f} signals")
