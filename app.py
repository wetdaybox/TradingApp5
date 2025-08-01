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
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def load_history(coin_id, vs, days):
    data = fetch_json(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        {"vs_currency": vs, "days": days}
    )["prices"]
    df = pd.DataFrame(data, columns=["ts","price"])
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
    return {
        "BTC": (j["bitcoin"]["usd"], j["bitcoin"]["usd_24h_change"]),
        "XRP": (j["ripple"]["btc"], None)
    }

# ── Backtests ──
def backtest_btc(df, rsi_th, tp_mult, sl_pct):
    df = df.copy()
    df["ema50"] = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    df["sma5"]  = df["price"].rolling(5).mean()
    df["sma20"] = df["price"].rolling(20).mean()
    delta = df["price"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    df["rsi"] = 100 - 100 / (1 + gain.rolling(RSI_WINDOW).mean()/loss.rolling(RSI_WINDOW).mean())
    df["vol"] = df["return"].rolling(VOL_WINDOW).std().fillna(0)

    wins = losses = 0
    for i in range(EMA_TREND, len(df)-1):
        p = df["price"].iat[i]
        if not (p>df["ema50"].iat[i] and df["sma5"].iat[i]>df["sma20"].iat[i] and df["rsi"].iat[i]<rsi_th):
            continue
        ret = df["return"].iat[i]; vol = df["vol"].iat[i]
        if ret < vol: continue
        drop = vol if ret<=2*vol else 2*vol
        tp   = drop*tp_mult/100 * p
        sl   = sl_pct/100 * p
        if df["price"].iat[i+1] > p: wins += 1
        else: losses += 1
    total = wins + losses
    return wins/total if total else 0.0

def backtest_xrp(df, mean_d, bounce_pct, sl_pct, min_bounce_pct):
    df = df.copy()
    df["mean"] = df["price"].rolling(mean_d).mean()
    df["vol"]  = df["return"].rolling(VOL_WINDOW).std().fillna(0)

    wins = losses = 0
    for i in range(mean_d, len(df)-1):
        p = df["price"].iat[i]; m = df["mean"].iat[i]
        gap_pct = (m-p)/p*100
        if not (p<m and gap_pct>=min_bounce_pct and df["vol"].iat[i]>df["vol"].iat[i-1]):
            continue
        tp = gap_pct/100 * p * (bounce_pct/100)
        sl = tp * sl_pct/100
        if df["price"].iat[i+1] >= p+tp: wins += 1
        else: losses += 1
    total = wins + losses
    return wins/total if total else 0.0

# ── Load data ──
btc_hist = load_history("bitcoin","usd",HISTORY_DAYS)
xrp_hist = load_history("ripple","btc",HISTORY_DAYS)
live     = load_live()
btc_p, btc_ch = live["BTC"]
xrp_p, _      = live["XRP"]

# ── Hyperparameter grids ──
btc_grid = [(rsi,tp,sl) for rsi in (65,70,75,80,85) for tp in (1.0,1.5,2.0) for sl in (0.5,1.0,2.0)]
xrp_grid = [(m,b,sl,mb) for m in (5,10,15) for b in (50,75,100) for sl in (25,50,75) for mb in (1.0,1.5)]

# ── Auto-tune BTC defaults ──
btc_default = next((cfg for cfg in btc_grid if backtest_btc(btc_hist,*cfg)>=0.70),
                   max(btc_grid, key=lambda c: backtest_btc(btc_hist,*c)))

# ── Auto-tune XRP defaults ──
xrp_default = next((cfg for cfg in xrp_grid if backtest_xrp(xrp_hist,*cfg)>=0.70),
                   max(xrp_grid, key=lambda c: backtest_xrp(xrp_hist,*c)))

# ── Sidebar Inputs ──
st.sidebar.title("💰 Investment Settings")
usd_alloc      = st.sidebar.number_input("Investment ($)",10.0,1e6,500.0,10.0)
user_min_order = st.sidebar.number_input("Min Order (BTC)",1e-6,1e-2,5e-4,1e-6,format="%.6f")

# ── Calculate MIN_ORDER ──
MIN_ORDER = max(user_min_order, (usd_alloc/GRID_MAX)/btc_p if btc_p else 0)
if btc_p:
    st.sidebar.caption(f"🔒 Min Order ≥ {MIN_ORDER:.6f} BTC (~${MIN_ORDER*btc_p:.2f})")
else:
    st.sidebar.caption(f"🔒 Min Order ≥ {MIN_ORDER:.6f} BTC")

# ── Display tuned defaults ──
rsi_th, tp_btc, sl_btc = btc_default
mean_xrp, bpct_xrp, sl_xrp, mb_pct = xrp_default
st.sidebar.markdown("### ⚙️ Tuned Defaults")
st.sidebar.write(f"**BTC:** RSI<{rsi_th}, TP×{tp_btc}, SL{sl_btc}%")
st.sidebar.write(f"**XRP:** Mean{mean_xrp}d, Bounce{bpct_xrp}%, SL{sl_xrp}%, MinDip{mb_pct}%")

# ── Grid calc ──
def compute_grid(price, drop, levels):
    bot  = price*(1-drop/100)
    step = (price-bot)/levels
    return bot,step

# ── BTC Bot ──
st.header("🟡 BTC/USDT Bot")
st.write(f"- Price: ${btc_p:.2f} | 24h Δ: {btc_ch:.2f}%")
vol14 = btc_hist["return"].rolling(VOL_WINDOW).std().iloc[-1]
ch    = btc_ch if btc_ch is not None else btc_hist["return"].iloc[-1]
drop  = (vol14 if ch<=2*vol14 else 2*vol14) if ch>=vol14 else None
if drop:
    st.write(f"- Reset drop: {drop:.2f}%")
    for L,label in zip((GRID_PRIMARY,GRID_FEWER,GRID_MORE),("Profitable","Fewer","More")):
        bot,step = compute_grid(btc_p,drop,L)
        per = (usd_alloc/btc_p)/L
        st.markdown(f"**{label}({L})** Lower:`{bot:.2f}` Upper:`{btc_p:.2f}` Step:`{step:.4f}` Per:`{per:.6f}` BTC {'✅' if per>=MIN_ORDER else '❌'}")
else:
    st.info("No grid reset")

# ── XRP Bot ──
st.header("🟣 XRP/BTC Bot")
st.write(f"- Price: {xrp_p:.6f} BTC")
hist = xrp_hist.copy()
sig = (hist["price"].iat[-1]<hist["price"].rolling(mean_xrp).mean().iat[-1]) and ((hist["return"].rolling(VOL_WINDOW).std().iat[-1])>(hist["return"].rolling(VOL_WINDOW).std().iat[-2])) and (((hist["price"].rolling(mean_xrp).mean().iat[-1]-hist["price"].iat[-1])/hist["price"].iat[-1]*100)>=mb_pct)
st.write("- Signal: ✅ Reset" if sig else "- Signal: ❌ None")
if sig:
    drop = bpct_xrp
    bot,step = compute_grid(xrp_p,drop,GRID_PRIMARY)
    per = (usd_alloc/btc_p)/GRID_PRIMARY
    st.markdown(f"**Primary({GRID_PRIMARY})** Lower:`{bot:.6f}` Upper:`{xrp_p:.6f}` Step:`{step:.8f}` Per:`{per:.6f}` BTC {'✅' if per>=MIN_ORDER else '❌'}")

# ── About ──
with st.expander("ℹ️ About"):
    st.markdown("""
    • Auto-tuned for ≥70% win rate over past 90 days (real data).  
    • Falls back to best if none meet threshold.  
    • BTC tests RSI up to 85; XRP enforces 1–1.5% min dip.  
    • Copy outputs into Crypto.com grid bot.
    """)
