# app.py - Crypto.com Exchange Grid Optimizer with Bespoke Grid Customization
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import time
from streamlit_autorefresh import st_autorefresh

# ── Auto-refresh every 5 min ──
st_autorefresh(interval=300_000, key="refresh")

# ── Constants ──
HISTORY_DAYS = 90
VOL_WINDOW = 14
RSI_WINDOW = 14
EMA_TREND = 50
MIN_VOLATILITY = 18
COINGECKO_DELAY = 3
EXCHANGE_MIN_ORDER_USD = 10.00  # Crypto.com Exchange requirement
USER_CRO_STAKE = 50_000  # Your CRO stake on Exchange

# Tier thresholds (Exchange-specific)
GOLD_TIER = 50_000  # Your tier
GRID_MAX = 100  # Max grids for Gold tier

# ── Page Setup ──
st.set_page_config(layout="centered")
st.title(f"🇬🇧 Crypto.com Exchange Grid Optimizer")
now = datetime.now(pytz.timezone("Europe/London"))
st.caption(f"Last updated: {now:%Y-%m-%d %H:%M %Z} | London")

# ── Exchange Benefits Display ──
st.sidebar.title("💷 Exchange Settings")
st.sidebar.success(f"Gold Tier: {USER_CRO_STAKE:,} CRO staked")
st.sidebar.subheader("Exchange Benefits")
st.sidebar.metric("Spot Trading Discount", "20%")
st.sidebar.metric("Maker Fees", "0.00%")
st.sidebar.metric("Min Order Value", "$10.00")
st.sidebar.caption("FCA Registered: Firm #906456")

# ── Currency Converter ──
st.sidebar.subheader("💱 Currency Conversion")
gbp_usd_rate = st.sidebar.number_input("GBP/USD Exchange Rate", 1.20, 1.50, 1.27, 0.01)
usd_alloc = st.sidebar.number_input("Investment (USD)", 10.0, 100000.0, 3000.0, 100.0)
gbp_alloc = usd_alloc / gbp_usd_rate

# Display allocation
st.sidebar.metric("Investment Value", f"£{gbp_alloc:,.2f}", f"${usd_alloc:,.2f}")

# ── Bespoke Grid Customization ──
st.sidebar.subheader("⚙️ Grid Customization")
enable_custom_grids = st.sidebar.checkbox("Enable Custom Grid Settings", value=False)

# ── Helpers ──
def fetch_json(url, params):
    time.sleep(COINGECKO_DELAY)
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.warning("⚠️ API rate limit approached. Using cached data.")
            return None
        raise

@st.cache_data(ttl=600)
def load_history(coin_id, vs, days):
    data = fetch_json(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        {"vs_currency": vs, "days": days}
    )
    if data is None:
        return pd.DataFrame()
    prices = data.get("prices", [])
    df = pd.DataFrame(prices, columns=["ts","price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change()*100
    return df.dropna()

@st.cache_data(ttl=300)
def load_live():
    j = fetch_json(
        "https://api.coingecko.com/api/v3/simple/price",
        {"ids":"bitcoin,ripple", "vs_currencies":"usd,btc", "include_24hr_change":"true"}
    )
    if j is None:
        return {
            "BTC": (st.session_state.get('last_btc_price', 0), 
                    st.session_state.get('last_btc_change', 0)),
            "XRP": (st.session_state.get('last_xrp_price', 0), None)
        }
    
    btc_price = j["bitcoin"]["usd"]
    btc_change = j["bitcoin"]["usd_24h_change"]
    xrp_price = j["ripple"]["btc"]
    
    st.session_state['last_btc_price'] = btc_price
    st.session_state['last_btc_change'] = btc_change
    st.session_state['last_xrp_price'] = xrp_price
    
    return {
        "BTC": (btc_price, btc_change),
        "XRP": (xrp_price, None)
    }

# ── Dual Currency Display Helper ──
def dual_currency(usd_amount, gbp_usd_rate):
    return f"${usd_amount:,.2f} | £{usd_amount/gbp_usd_rate:,.2f}"

# ── Backtests ──
def backtest_btc(df, rsi_th, tp_mult, sl_pct):
    df = df.copy()
    df["ema50"] = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    df["sma5"] = df["price"].rolling(5).mean()
    df["sma20"] = df["price"].rolling(20).mean()
    delta = df["price"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    df["rsi"] = 100 - 100 / (1 + gain.rolling(RSI_WINDOW).mean()/loss.rolling(RSI_WINDOW).mean())
    df["vol"] = df["return"].rolling(VOL_WINDOW).std().fillna(0)

    wins = losses = 0
    for i in range(EMA_TREND, len(df)-1):
        if df["vol"].iat[i] < MIN_VOLATILITY:
            continue
            
        p = df["price"].iat[i]
        if not (p>df["ema50"].iat[i] and df["sma5"].iat[i]>df["sma20"].iat[i] and df["rsi"].iat[i]<rsi_th):
            continue
            
        if abs(df["return"].iat[i]) > 3 * df["vol"].iat[i]:
            continue
            
        ret = df["return"].iat[i]; vol = df["vol"].iat[i]
        if ret < vol: continue
        drop = vol if ret<=2*vol else 2*vol
        tp = drop*tp_mult/100 * p
        sl = sl_pct/100 * p
        if df["price"].iat[i+1] > p: wins += 1
        else: losses += 1
    total = wins + losses
    return wins/total if total else 0.0

def backtest_xrp(df, mean_d, bounce_pct, sl_pct, min_bounce_pct):
    df = df.copy()
    df["mean"] = df["price"].rolling(mean_d).mean()
    df["vol"] = df["return"].rolling(VOL_WINDOW).std().fillna(0)
    df["sma5"] = df["price"].rolling(5).mean()
    df["sma20"] = df["price"].rolling(20).mean()

    wins = losses = 0
    for i in range(mean_d, len(df)-1):
        if df["vol"].iat[i] < MIN_VOLATILITY:
            continue
            
        p = df["price"].iat[i]; m = df["mean"].iat[i]
        gap_pct = (m-p)/p*100
        momentum_ok = df["sma5"].iat[i] > df["sma20"].iat[i]
        if not (p<m and gap_pct>=min_bounce_pct and df["vol"].iat[i]>df["vol"].iat[i-1] and momentum_ok):
            continue
            
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
live = load_live()
btc_p, btc_ch = live["BTC"]
xrp_p, _ = live["XRP"]

# ── Hyperparameter grids ──
btc_grid = [(rsi,tp,sl) for rsi in (65,70,75,80,85) for tp in (1.0,1.5,2.0) for sl in (0.5,1.0,2.0)]
xrp_grid = [(m,b,sl,mb) for m in (5,10,15) for b in (50,75,100) for sl in (25,50,75) for mb in (1.0,1.5)]

# ── Auto-tune defaults ──
btc_default = next((cfg for cfg in btc_grid if backtest_btc(btc_hist,*cfg)>=0.70),
                   max(btc_grid, key=lambda c: backtest_btc(btc_hist,*c)))

xrp_default = next((cfg for cfg in xrp_grid if backtest_xrp(xrp_hist,*cfg)>=0.70),
                   max(xrp_grid, key=lambda c: backtest_xrp(xrp_hist,*c)))

# ── Asset Allocation ──
st.sidebar.subheader("📊 Asset Allocation")
btc_percent = st.sidebar.slider("BTC Allocation (%)", 0, 100, 70)
xrp_percent = 100 - btc_percent
btc_alloc_usd = usd_alloc * btc_percent / 100
xrp_alloc_usd = usd_alloc * xrp_percent / 100

# Display allocation in both currencies
btc_alloc_gbp = btc_alloc_usd / gbp_usd_rate
xrp_alloc_gbp = xrp_alloc_usd / gbp_usd_rate
st.sidebar.caption(f"BTC: ${btc_alloc_usd:,.2f} | £{btc_alloc_gbp:,.2f}")
st.sidebar.caption(f"XRP: ${xrp_alloc_usd:,.2f} | £{xrp_alloc_gbp:,.2f}")

# ── Exchange Grid Calculation ──
def calculate_max_grids(investment_usd):
    """Exchange-compatible grid calculation"""
    max_possible = int(investment_usd / EXCHANGE_MIN_ORDER_USD)
    return min(max_possible, GRID_MAX)

def compute_grid(price, drop, levels):
    bot = price * (1 - drop/100)
    step = (price - bot) / levels
    return bot, step

def optimize_grid_levels(investment_usd, volatility):
    """Auto-select grid density based on volatility and investment"""
    max_possible = calculate_max_grids(investment_usd)
    
    # Volatility-based tuning
    if volatility < 15:   # Low volatility
        return min(50, max_possible)  # Denser grids
    elif volatility > 30: # High volatility
        return min(25, max_possible)  # Wider spacing
    else:                 # Moderate volatility
        return min(35, max_possible)  # Balanced approach

# ── Strategy Display ──
rsi_th, tp_btc, sl_btc = btc_default
mean_xrp, bpct_xrp, sl_xrp, mb_pct = xrp_default
st.sidebar.markdown("### ⚙️ Tuned Parameters")
st.sidebar.write(f"**BTC:** RSI<{rsi_th}, TP×{tp_btc}, SL{sl_btc}%")
st.sidebar.write(f"**XRP:** Mean{mean_xrp}d, Bounce{bpct_xrp}%, SL{sl_xrp}%, MinDip{mb_pct}%")

# ── BTC Grids ──
st.header("🟡 BTC/USD Exchange Grids")
st.write(f"- Price: ${btc_p:,.2f} | {dual_currency(btc_p, gbp_usd_rate)}")
st.write(f"- 24h Change: {btc_ch:.2f}%")

vol14 = btc_hist["return"].rolling(VOL_WINDOW).std().iloc[-1]
ch = btc_ch if btc_ch is not None else btc_hist["return"].iloc[-1]

if vol14 < MIN_VOLATILITY:
    st.warning(f"⚠️ Low Volatility ({vol14:.1f}% < {MIN_VOLATILITY}%)")
else:
    drop = (vol14 if ch <= 2*vol14 else 2*vol14) if ch >= vol14 else None
    if drop:
        # Calculate max possible grids
        max_grids = calculate_max_grids(btc_alloc_usd)
        
        # Bespoke customization option
        if enable_custom_grids:
            st.subheader("🔧 Custom Grid Configuration")
            custom_grids = st.slider(
                "Select BTC Grid Levels", 
                5, 
                min(max_grids, GRID_MAX),
                min(20, max_grids),  # Default to 20 or max possible
                key="btc_grids"
            )
            grids_to_use = custom_grids
            st.write(f"Custom grids selected: {grids_to_use}")
        else:
            # Auto-optimized approach
            optimized_grids = optimize_grid_levels(btc_alloc_usd, vol14)
            grids_to_use = optimized_grids
            st.write(f"System-recommended grids: {optimized_grids}")
        
        # Calculate grid parameters
        bot, step = compute_grid(btc_p, drop, grids_to_use)
        order_value_usd = btc_alloc_usd / grids_to_use
        
        # Check if meets minimum
        meets_min = order_value_usd >= EXCHANGE_MIN_ORDER_USD
        
        st.subheader(f"🔷 Grid Configuration: {grids_to_use} Levels")
        
        col1, col2 = st.columns(2)
        col1.metric("Lower Bound", f"${bot:,.2f}", f"£{bot/gbp_usd_rate:,.2f}")
        col2.metric("Upper Bound", f"${btc_p:,.2f}", f"£{btc_p/gbp_usd_rate:,.2f}")
        col1.metric("Price Step", f"${step:,.2f}", f"£{step/gbp_usd_rate:,.2f}")
        
        # Highlight if order value meets requirement
        order_status = "✅ Meets minimum" if meets_min else f"❌ Below ${EXCHANGE_MIN_ORDER_USD} min"
        col2.metric("Order Value", f"${order_value_usd:,.2f}", order_status)
        
        st.code(f"""Exchange Setup:
Currency Pair: BTC/USD
Price Range: ${bot:,.2f} - ${btc_p:,.2f}
Grid Levels: {grids_to_use}
Order Value: ${order_value_usd:,.2f}""")
        
        # Warning if below minimum
        if not meets_min:
            st.error(f"⚠️ Order value (${order_value_usd:,.2f}) below minimum requirement (${EXCHANGE_MIN_ORDER_USD}). "
                     f"Please reduce grid levels or increase investment.")
    else:
        st.info("No grid reset recommended")

# ── XRP Grids ──
st.header("🟣 XRP/BTC Exchange Grids")
st.write(f"- Price: {xrp_p:.6f} BTC")

# Compute technicals for XRP
xrp_hist["sma5"] = xrp_hist["price"].rolling(5).mean()
xrp_hist["sma20"] = xrp_hist["price"].rolling(20).mean()
vol14_xrp = xrp_hist["return"].rolling(VOL_WINDOW).std().iloc[-1]
last_return = xrp_hist["return"].iloc[-1]

# Signal components
price_below_ma = xrp_p < xrp_hist["price"].rolling(mean_xrp).mean().iloc[-1]
vol_increasing = vol14_xrp > xrp_hist["return"].rolling(VOL_WINDOW).std().iloc[-2]
gap_pct = (xrp_hist["price"].rolling(mean_xrp).mean().iloc[-1] - xrp_p) / xrp_p * 100  # Fixed syntax
momentum_ok = xrp_hist["sma5"].iloc[-1] > xrp_hist["sma20"].iloc[-1]

sig = price_below_ma and vol_increasing and (gap_pct >= mb_pct) and momentum_ok

# Display signal status
st.write("### Signal Diagnostics")
col1, col2 = st.columns(2)
col1.write(f"**Price < MA:** {'✅' if price_below_ma else '❌'}")
col2.write(f"**Gap from MA:** {gap_pct:.2f}% (Min: {mb_pct}%)")
col1.write(f"**Vol Increasing:** {'✅' if vol_increasing else '❌'}")
col2.write(f"**Momentum:** {'✅' if momentum_ok else '❌'}")
col1.write(f"**Volatility:** {vol14_xrp:.2f}%")
col2.write(f"**Min Volatility:** {MIN_VOLATILITY}%")

if vol14_xrp < MIN_VOLATILITY:
    st.warning(f"⚠️ Low Volatility ({vol14_xrp:.1f}% < {MIN_VOLATILITY}%)")
elif not sig:
    st.info("❌ Mean-reversion signal not active")
else:
    # Calculate max possible grids
    max_grids = calculate_max_grids(xrp_alloc_usd)
    
    # Bespoke customization option
    if enable_custom_grids:
        st.subheader("🔧 Custom Grid Configuration")
        custom_grids = st.slider(
            "Select XRP Grid Levels", 
            5, 
            min(max_grids, GRID_MAX),
            min(15, max_grids),  # Default to 15 or max possible
            key="xrp_grids"
        )
        grids_to_use = custom_grids
        st.write(f"Custom grids selected: {grids_to_use}")
    else:
        # Auto-optimized approach
        optimized_grids = optimize_grid_levels(xrp_alloc_usd, vol14_xrp)
        grids_to_use = optimized_grids
        st.write(f"System-recommended grids: {optimized_grids}")
    
    # Calculate grid parameters
    bot, step = compute_grid(xrp_p, bpct_xrp, grids_to_use)
    
    # Calculate BTC equivalent for order size
    btc_value_per_order = (xrp_alloc_usd / btc_p) / grids_to_use
    usd_value_per_order = xrp_alloc_usd / grids_to_use
    
    # Check if meets minimum
    meets_min = usd_value_per_order >= EXCHANGE_MIN_ORDER_USD
    
    st.subheader(f"🔷 Grid Configuration: {grids_to_use} Levels")
    
    col1, col2 = st.columns(2)
    col1.metric("Lower Bound", f"{bot:.6f} BTC")
    col2.metric("Upper Bound", f"{xrp_p:.6f} BTC")
    col1.metric("Price Step", f"{step:.8f} BTC")
    
    # Highlight if order value meets requirement
    order_status = "✅ Meets minimum" if meets_min else f"❌ Below ${EXCHANGE_MIN_ORDER_USD} min"
    col2.metric("Order Value", f"${usd_value_per_order:,.2f}", order_status)
    
    st.code(f"""Exchange Setup:
Currency Pair: XRP/BTC
Price Range: {bot:.6f} - {xrp_p:.6f} BTC
Grid Levels: {grids_to_use}
Order Size: {btc_value_per_order:.6f} BTC""")
    
    # Warning if below minimum
    if not meets_min:
        st.error(f"⚠️ Order value (${usd_value_per_order:,.2f}) below minimum requirement (${EXCHANGE_MIN_ORDER_USD}). "
                 f"Please reduce grid levels or increase investment.")

# ── UK Compliance Notice ──
st.caption("""
**UK Regulatory Disclosure:**  
Crypto.com Exchange UK Ltd (FRN: 906456) is registered with the Financial Conduct Authority.  
Cryptocurrency investments are high risk. Capital at risk. Past performance ≠ future results.
""")
