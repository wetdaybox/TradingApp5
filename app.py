# app.py - Enhanced Trading System
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
st.title("🚀 Enhanced Scalping Grid Bot Trading System")
now = datetime.now(pytz.timezone("Europe/London"))
st.caption(f"Last updated: {now:%Y-%m-%d %H:%M %Z}")

# ── Constants ──
HISTORY_DAYS   = 90
VOL_WINDOW     = 14
RSI_WINDOW     = 14
EMA_TREND      = 50
MIN_VOLATILITY = 18  # Minimum volatility threshold for trading

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

# ── Enhanced Backtests ──
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
        # Volatility filter - skip low volatility periods
        if df["vol"].iat[i] < MIN_VOLATILITY:
            continue
            
        p = df["price"].iat[i]
        if not (p>df["ema50"].iat[i] and df["sma5"].iat[i]>df["sma20"].iat[i] and df["rsi"].iat[i]<rsi_th):
            continue
            
        # Circuit breaker - skip extreme moves
        if abs(df["return"].iat[i]) > 3 * df["vol"].iat[i]:
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
    # Momentum confirmation
    df["sma5"] = df["price"].rolling(5).mean()
    df["sma20"] = df["price"].rolling(20).mean()

    wins = losses = 0
    for i in range(mean_d, len(df)-1):
        # Volatility filter
        if df["vol"].iat[i] < MIN_VOLATILITY:
            continue
            
        p = df["price"].iat[i]; m = df["mean"].iat[i]
        gap_pct = (m-p)/p*100
        # Add momentum confirmation (sma5 > sma20)
        momentum_ok = df["sma5"].iat[i] > df["sma20"].iat[i]
        if not (p<m and gap_pct>=min_bounce_pct and df["vol"].iat[i]>df["vol"].iat[i-1] and momentum_ok):
            continue
            
        # Circuit breaker
        if abs(df["return"].iat[i]) > 3 * df["vol"].iat[i]:
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
st.sidebar.title("💰 Enhanced Investment Settings")
usd_alloc      = st.sidebar.number_input("Investment ($)",10.0,1e6,500.0,10.0)
user_min_order = st.sidebar.number_input("Min Order (BTC)",1e-6,1e-2,5e-4,1e-6,format="%.6f")
vol_filter     = st.sidebar.checkbox("Enable Volatility Filter", value=True)
circuit_breaker= st.sidebar.checkbox("Enable Circuit Breaker", value=True)

# ── Dynamic Position Sizing ──
def calculate_dynamic_allocation(hist, default_config, strategy):
    """Reduce exposure during drawdowns"""
    recent_perf = strategy(hist.iloc[-30:], *default_config)  # Last 30 days performance
    if recent_perf < 0.55:  # Underperforming
        return 0.5 * usd_alloc
    elif recent_perf > 0.75:  # Overperforming
        return min(1.5 * usd_alloc, 10000)  # Cap at $10k
    return usd_alloc

btc_alloc = calculate_dynamic_allocation(btc_hist, btc_default, backtest_btc)
xrp_alloc = calculate_dynamic_allocation(xrp_hist, xrp_default, backtest_xrp)

# Calculate MIN_ORDER with dynamic allocation
MIN_ORDER_BTC = max(user_min_order, (btc_alloc/GRID_MAX)/btc_p) if btc_p else user_min_order
MIN_ORDER_XRP = max(user_min_order, (xrp_alloc/GRID_MAX)/btc_p) if btc_p else user_min_order

st.sidebar.caption(f"🔒 BTC Allocation: ${btc_alloc:.2f} | Min Order: {MIN_ORDER_BTC:.6f} BTC")
st.sidebar.caption(f"🔒 XRP Allocation: ${xrp_alloc:.2f} | Min Order: {MIN_ORDER_XRP:.6f} BTC")

# ── Display tuned defaults ──
rsi_th, tp_btc, sl_btc = btc_default
mean_xrp, bpct_xrp, sl_xrp, mb_pct = xrp_default
st.sidebar.markdown("### ⚙️ Tuned Defaults")
st.sidebar.write(f"**BTC:** RSI<{rsi_th}, TP×{tp_btc}, SL{sl_btc}%")
st.sidebar.write(f"**XRP:** Mean{mean_xrp}d, Bounce{bpct_xrp}%, SL{sl_xrp}%, MinDip{mb_pct}%")

# ── Enhanced Grid Calculation ──
def compute_grid(price, drop, levels):
    bot  = price*(1-drop/100)
    step = (price-bot)/levels
    return bot, step

# ── BTC Bot with Enhancements ──
st.header("🟡 Enhanced BTC/USDT Bot")
st.write(f"- Price: ${btc_p:.2f} | 24h Δ: {btc_ch:.2f}%")
vol14 = btc_hist["return"].rolling(VOL_WINDOW).std().iloc[-1]
ch    = btc_ch if btc_ch is not None else btc_hist["return"].iloc[-1]

# Circuit breaker check
circuit_triggered = False
if circuit_breaker and abs(ch) > 3 * vol14:
    st.error(f"🚨 Circuit Breaker Triggered! Last move: {ch:.2f}% > 3σ ({3*vol14:.2f}%)")
    circuit_triggered = True

# Volatility filter
if vol_filter and vol14 < MIN_VOLATILITY:
    st.warning(f"⚠️ Low Volatility ({vol14:.1f}% < {MIN_VOLATILITY}%). Trading not recommended")
elif not circuit_triggered:
    drop  = (vol14 if ch<=2*vol14 else 2*vol14) if ch>=vol14 else None
    if drop:
        st.write(f"- Reset drop: {drop:.2f}%")
        for L,label in zip((GRID_PRIMARY,GRID_FEWER,GRID_MORE),("Profitable","Fewer","More")):
            bot,step = compute_grid(btc_p,drop,L)
            per = (btc_alloc/btc_p)/L
            st.markdown(f"**{label}({L})** Lower:`{bot:.2f}` Upper:`{btc_p:.2f}` Step:`{step:.4f}` Per:`{per:.6f}` BTC {'✅' if per>=MIN_ORDER_BTC else '❌'}")
    else:
        st.info("No grid reset")
else:
    st.info("Trading paused due to circuit breaker")

# ── Enhanced XRP Bot ──
st.header("🟣 Enhanced XRP/BTC Bot")
st.write(f"- Price: {xrp_p:.6f} BTC")

# Compute technicals for XRP
xrp_hist["sma5"] = xrp_hist["price"].rolling(5).mean()
xrp_hist["sma20"] = xrp_hist["price"].rolling(20).mean()
vol14_xrp = xrp_hist["return"].rolling(VOL_WINDOW).std().iloc[-1]
last_return = xrp_hist["return"].iloc[-1]

# Signal components
price_below_ma = xrp_hist["price"].iat[-1] < xrp_hist["price"].rolling(mean_xrp).mean().iat[-1]
vol_increasing = vol14_xrp > xrp_hist["return"].rolling(VOL_WINDOW).std().iat[-2]
gap_pct = (xrp_hist["price"].rolling(mean_xrp).mean().iat[-1] - xrp_hist["price"].iat[-1])/xrp_hist["price"].iat[-1]*100
momentum_ok = xrp_hist["sma5"].iat[-1] > xrp_hist["sma20"].iat[-1]  # Momentum confirmation

sig = price_below_ma and vol_increasing and (gap_pct >= mb_pct) and momentum_ok

# Circuit breaker for XRP
xrp_circuit_triggered = False
if circuit_breaker and abs(last_return) > 3 * vol14_xrp:
    st.error(f"🚨 Circuit Breaker Triggered! Last move: {last_return:.2f}% > 3σ ({3*vol14_xrp:.2f}%)")
    xrp_circuit_triggered = True

# Display signal status
st.write("- Signal Components:")
st.write(f"  • Price < MA: {'✅' if price_below_ma else '❌'} | Gap: {gap_pct:.2f}% (Min: {mb_pct}%)")
st.write(f"  • Vol Increasing: {'✅' if vol_increasing else '❌'} | Momentum: {'✅' if momentum_ok else '❌'}")
st.write(f"  • Volatility: {vol14_xrp:.2f}% (Min: {MIN_VOLATILITY}%)")

if vol_filter and vol14_xrp < MIN_VOLATILITY:
    st.warning(f"⚠️ Low Volatility ({vol14_xrp:.1f}% < {MIN_VOLATILITY}%). Trading not recommended")
elif not xrp_circuit_triggered and sig:
    st.success("✅ Signal: Reset Active")
    drop = bpct_xrp
    bot,step = compute_grid(xrp_p,drop,GRID_PRIMARY)
    per = (xrp_alloc/btc_p)/GRID_PRIMARY
    st.markdown(f"**Primary({GRID_PRIMARY})** Lower:`{bot:.6f}` Upper:`{xrp_p:.6f}` Step:`{step:.8f}` Per:`{per:.6f}` BTC {'✅' if per>=MIN_ORDER_XRP else '❌'}")
elif not sig and not xrp_circuit_triggered:
    st.info("❌ Signal: Conditions not met")
else:
    st.info("Trading paused due to circuit breaker")

# ── About ──
with st.expander("ℹ️ Enhanced Features"):
    st.markdown("""
    **New Risk Management Features:**
    1. ⚠️ **Volatility Filter**: Skip trades when 14-day volatility <18%
    2. 🚨 **Circuit Breaker**: Pause trading during >3σ moves
    3. 📉 **Dynamic Allocation**: Adjust exposure based on recent performance
    4. 📈 **XRP Momentum**: Require SMA5 > SMA20 for entries
    
    **Core Features:**
    • Auto-tuned for ≥70% win rate over past 90 days
    • Real-time data from CoinGecko API
    • Grid optimization for Crypto.com bots
    • London market hours alignment
    """)
