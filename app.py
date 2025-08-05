# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# ── Auto-refresh every 60 s ──
st_autorefresh(interval=60_000, key="refresh")

# ── Page setup ──
st.set_page_config(layout="centered")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
now = datetime.now(pytz.timezone("Europe/London"))
st.caption(f"Last updated: {now:%Y-%m-%d %H:%M %Z}")

# ── Constants ──
HISTORY_DAYS   = 90
VOL_WINDOW     = 14
RSI_WINDOW     = 14
EMA_TREND      = 50

GRID_PRIMARY   = 20
GRID_FEWER     = 10
GRID_MORE      = 30
GRID_MAX       = 30

# ── Helpers ──
def fetch_json(url, params):
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

@st.cache_data(ttl=600)
def load_history(coin_id, vs, days):
    data = fetch_json(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        {"vs_currency": vs, "days": days}
    )
    if not data or "prices" not in data:
        st.warning(f"⚠️ Failed to load history for {coin_id}")
        return pd.DataFrame(columns=["price","return"])
    df = pd.DataFrame(data["prices"], columns=["ts","price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change()*100
    return df.dropna()

@st.cache_data(ttl=60)
def load_live():
    j = fetch_json(
        "https://api.coingecko.com/api/v3/simple/price",
        {"ids":"bitcoin,ripple","vs_currencies":"usd,btc","include_24hr_change":"true"}
    )
    if not j:
        st.warning("⚠️ Failed to load live prices.")
        return {"BTC":(np.nan,None), "XRP":(np.nan,None)}
    return {
        "BTC": (j["bitcoin"]["usd"], j["bitcoin"].get("usd_24h_change")),
        "XRP": (j["ripple"]["btc"], None)
    }

# ── Load data ──
btc_hist = load_history("bitcoin","usd",HISTORY_DAYS)
xrp_hist = load_history("ripple","btc",HISTORY_DAYS)
live     = load_live()
btc_p, btc_ch = live["BTC"]
xrp_p, _      = live["XRP"]

usd_alloc      = st.sidebar.number_input("Investment ($)",10.0,1e6,500.0,10.0)
user_min_order = st.sidebar.number_input("Min Order (BTC)",1e-6,1e-2,5e-4,1e-6,format="%.6f")
MIN_ORDER = max(user_min_order, (usd_alloc/GRID_MAX)/ (btc_p or 1))

# ── Manual override injection ──
override = st.sidebar.checkbox("Manual Grid Override", value=False)
manual_b = st.sidebar.number_input("BTC/USDT Grids", 2, GRID_MAX, GRID_PRIMARY) if override else None
manual_x = st.sidebar.number_input("XRP/BTC Grids",   2, GRID_MAX, GRID_PRIMARY) if override else None

# ── Bot display ──
for label, price, hist, manual in [
    ("🟡 BTC/USDT", btc_p, btc_hist, manual_b),
    ("🟣 XRP/BTC",   xrp_p, xrp_hist, manual_x)
]:
    st.header(f"{label} Bot")
    if price is None or np.isnan(price) or hist.empty:
        st.info("Data unavailable—waiting for both history and live price.")
        continue

    vol = hist["return"].rolling(VOL_WINDOW).std().iloc[-1]
    ch  = btc_ch if label.startswith("🟡") else hist["return"].iloc[-1]
    drop = (vol if ch<=2*vol else 2*vol) if ch>=vol else 0

    levels = manual if manual is not None else GRID_PRIMARY
    bot = price*(1-drop/100)
    step = (price-bot)/levels
    per  = (usd_alloc/price)/levels

    st.metric("Grids", levels)
    st.metric("Lower", f"{bot:.6f}")
    st.metric("Upper", f"{price:.6f}")
    st.metric("Step",  f"{step:.6f}")
    st.metric("Per-Grid", f"{per:.6f} BTC {'✅' if per>=MIN_ORDER else '❌'}")

    st.info("⚠️ Waiting to deploy when conditions are met.")

# ── About ──
with st.expander("ℹ️ About & Features"):
    st.markdown("""
    • Manual override lets you choose any grid count without assumption.  
    • Defaults to 20 levels if no override.  
    • Historical 90-day volatility drives reset drop calculation.  
    • Copy the displayed ranges into Crypto.com’s grid bot.
    """)
