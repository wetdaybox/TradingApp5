import streamlit as st
import requests, pandas as pd, numpy as np
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# ── Auto-refresh every 60 s ──
st_autorefresh(interval=60_000, key="datarefresh")
st.set_page_config(
    page_title="Infinite Scalping Grid Bot Trading System",
    layout="centered"
)

# ── Sidebar: Settings ──
st.sidebar.title("⚙️ Global Settings")
HISTORY_DAYS = st.sidebar.slider("History Window (days)", 30, 180, 90)
VOL_WINDOW   = st.sidebar.slider("Volatility Window (days)", 7, 30, 14)

st.sidebar.markdown("### 🟡 BTC/USDT Strategy")
RSI_OVER      = st.sidebar.slider("RSI Overbought Threshold", 60, 90, 75)
TP_MULT_BTC   = st.sidebar.slider("BTC TP Multiplier", 0.5, 2.0, 1.5, 0.1)
SL_PCT_BTC    = st.sidebar.slider("BTC SL % below bottom", 0.5, 5.0, 1.0, 0.1)

st.sidebar.markdown("### 🟣 XRP/BTC Strategy")
XRP_MEAN_D    = st.sidebar.slider("Mean Window (days)", 5, 20, 10)
TP_MULT_XRP   = st.sidebar.slider("XRP TP Multiplier", 0.5, 2.0, 1.0, 0.1)
SL_PCT_XRP    = st.sidebar.slider("XRP SL % of bounce", 10, 100, 50, 5)
MIN_BOUNCE_PCT= st.sidebar.slider("Min Gap to Mean (%)", 0.1, 1.0, 0.3, 0.1)

st.sidebar.markdown("### 💰 Investment")
usd_alloc     = st.sidebar.number_input("Investment in USD", min_value=1.0, value=100.0, step=1.0)
user_min_order= st.sidebar.number_input("Manual Min Order BTC", 1e-6, 1e-3, 5e-4, 1e-6)

GRID_MIN, GRID_MAX = 1, 30

# ── Caching fallback ──
_last_btc_df = None
_last_live   = None

def fetch_json(url, params):
    r = requests.get(url, params=params, timeout=10)
    if r.status_code == 429:
        raise requests.exceptions.HTTPError("429")
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def get_btc_history():
    global _last_btc_df
    try:
        data = fetch_json(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            {"vs_currency":"usd","days":HISTORY_DAYS}
        )["prices"]
        df = pd.DataFrame(data, columns=["ts","price"])
        df["date"] = pd.to_datetime(df["ts"],unit="ms")
        df = df.set_index("date").resample("D").last().dropna()
        df["return"] = df["price"].pct_change()*100
        df["vol"]    = df["return"].rolling(VOL_WINDOW).std()
        df["sma5"]   = df["price"].rolling(5).mean()
        df["sma20"]  = df["price"].rolling(20).mean()
        df["ema50"]  = df["price"].ewm(span=50,adjust=False).mean()
        delta       = df["price"].diff()
        gain        = delta.clip(lower=0)
        loss        = -delta.clip(upper=0)
        avg_gain    = gain.rolling(14).mean()
        avg_loss    = loss.rolling(14).mean().replace(0,np.nan)
        df["rsi"]   = 100 - 100/(1 + avg_gain/avg_loss)
        _last_btc_df = df.dropna()
        return _last_btc_df
    except requests.exceptions.HTTPError:
        st.warning("Rate-limited; using last BTC cache.")
        return _last_btc_df or pd.DataFrame()

@st.cache_data(ttl=60)
def get_live():
    global _last_live
    try:
        j = fetch_json(
            "https://api.coingecko.com/api/v3/simple/price",
            {"ids":"bitcoin,ripple","vs_currencies":"usd,btc","include_24hr_change":"true"}
        )
        _last_live = {
            "BTC": (j["bitcoin"]["usd"], j["bitcoin"].get("usd_24h_change",0)),
            "XRP": (j["ripple"]["btc"], None)
        }
        return _last_live
    except requests.exceptions.HTTPError:
        st.warning("Rate-limited; using last live cache.")
        return _last_live or {"BTC":(None,None),"XRP":(None,None)}

def compute_grid(top, drop, levels):
    bot = top*(1-drop/100)
    step= (top-bot)/levels
    return bot, step

# ── Load data ──
btc_hist = get_btc_history()
live     = get_live()
btc_p, btc_ch = live["BTC"]
xrp_p, _      = live["XRP"]

# USD→BTC
INV_BTC = (usd_alloc/btc_p) if btc_p else 0
# Auto-min order
usd_per_order       = usd_alloc/GRID_MAX
auto_min_order_btc  = usd_per_order/btc_p if btc_p else 0
MIN_ORDER           = max(user_min_order, auto_min_order_btc)
st.sidebar.caption(f"🔒 Min Order ≥ {MIN_ORDER:.6f} BTC (~${MIN_ORDER*btc_p:.2f})")

# ── Simulate XRP history ──
def get_xrp_history():
    np.random.seed(42)
    base = 0.02 + np.cumsum(np.random.normal(0,0.0015,len(btc_hist)))
    df = pd.DataFrame({"price":base}, index=btc_hist.index)
    df["return"] = df["price"].pct_change()*100
    df["vol"]    = df["return"].rolling(VOL_WINDOW).std()
    df["mean"]   = df["price"].rolling(XRP_MEAN_D).mean()
    df["signal"] = (
        (df["price"] < df["mean"]) &
        (df["vol"] > df["vol"].shift(1)) &
        ((df["mean"]-df["price"])/df["price"]*100 > MIN_BOUNCE_PCT)
    )
    return df.dropna()

xrp_hist = get_xrp_history()

# ── Header ──
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
now = datetime.now(pytz.timezone("Europe/London"))
st.caption(f"Last updated: {now:%Y-%m-%d %H:%M %Z}")

tab1, tab2 = st.tabs(["🟡 BTC/USDT", "🟣 XRP/BTC"])

# ── BTC Tab ──
with tab1:
    if btc_hist.empty or not btc_p:
        st.error("BTC data unavailable.")
    else:
        L = btc_hist.iloc[-1]
        mod, strg = L["vol"], 2*L["vol"]
        st.markdown(
            f"- **Price:** ${btc_p:.2f}  \n"
            f"- **24 h Δ:** {btc_ch:.2f}%  \n"
            f"- **Vol(14 d):** {mod:.2f}%  \n"
            f"- **RSI:** {L['rsi']:.1f}"
        )
        drop, status = None, "No grid recommended"
        if btc_ch >= mod:
            drop = mod if btc_ch<=strg else strg
            status = "Moderate reset" if btc_ch<=strg else "Strong reset"
        st.markdown(f"- **Signal:** {status}")
        ok = (L["price"]>L["ema50"] and L["sma5"]>L["sma20"] and L["rsi"]<RSI_OVER)
        st.markdown(f"- **Filters OK:** {'✅' if ok else '❌'}")
        if drop and ok:
            # backtest to pick L
            cond = (
                (btc_hist["return"]>=mod)&
                (btc_hist["price"]>btc_hist["ema50"])&
                (btc_hist["sma5"]>btc_hist["sma20"])&
                (btc_hist["rsi"]<RSI_OVER)
            )
            trades = cond.sum()
            wins   = ((btc_hist["price"].shift(-1)>btc_hist["price"])&cond).sum()
            wr     = wins/trades if trades else 0
            scores = [wr*(drop/L) for L in range(GRID_MIN,GRID_MAX+1)]
            opt    = int(np.argmax(scores))+GRID_MIN
            bot,step = compute_grid(btc_p,drop,opt)
            tp_price = bot + (btc_p-bot)*TP_MULT_BTC
            sl_price = bot*(1-SL_PCT_BTC/100)
            per_btc  = INV_BTC/opt
            per_usd  = per_btc*btc_p
            st.subheader("📐 Grid Recommendation")
            st.markdown(
                f"- **Upper:** `{btc_p:.2f}`  \n"
                f"- **Lower:** `{bot:.2f}`  \n"
                f"- **Levels:** `{opt}`  \n"
                f"- **Step:** `{step:.4f}`  \n"
                f"- **TP:** `{tp_price:.2f}`  \n"
                f"- **SL:** `{sl_price:.2f}`  \n"
                f"- **Per Order:** `{per_btc:.6f}` BTC (`${per_usd:.2f}`) "
                f"{'✅' if per_btc>=MIN_ORDER else '❌'}"
            )
            with st.expander("📋 Copyable Summary"):
                summ = (
                    f"Upper: {btc_p:.2f}\nLower: {bot:.2f}\n"
                    f"Levels: {opt}\nTP: {tp_price:.2f}\n"
                    f"SL: {sl_price:.2f}\n"
                    f"Per Order: {per_btc:.6f} BTC (${per_usd:.2f})"
                )
                st.code(summ, language="text")
        else:
            st.info("No grid recommended at this time.")

# ── XRP Tab ──
with tab2:
    if xrp_hist.empty or not xrp_p:
        st.error("XRP data unavailable.")
    else:
        L = xrp_hist.iloc[-1]
        st.markdown(
            f"- **Price:** {xrp_p:.6f} BTC  \n"
            f"- **Mean:** {L['mean']:.6f} BTC  \n"
            f"- **Vol(14 d):** {L['vol']:.2f}%"
        )
        sig = L["signal"]
        st.markdown(f"- **Signal:** {'✅ Reset' if sig else '❌ None'}")
        if sig:
            # compute grid levels
            df2 = xrp_hist.copy()
            df2["win"] = ((df2["price"].shift(-1)>df2["mean"]) & df2["signal"])
            wr2 = df2["win"].sum()/df2["signal"].sum() if df2["signal"].sum() else 0
            scores2 = [wr2*(L["vol"]/L2) for L2 in range(GRID_MIN,GRID_MAX+1)]
            opt2    = int(np.argmax(scores2))+GRID_MIN
            top     = L["mean"]
            bot2    = xrp_p*(1-L["vol"]/100)
            step2   = (top-bot2)/opt2
            # TP & SL based on bounce target
            bounce  = (top - xrp_p)*(TP_MULT_XRP/1)*(1)  # user TP mult
            tp2     = xrp_p + bounce
            sl_amt  = bounce*(SL_PCT_XRP/100)
            per_btc2= INV_BTC/opt2
            per_usd2= per_btc2*btc_p
            st.subheader("📐 Grid Recommendation")
            st.markdown(
                f"- **Upper:** `{top:.6f}` BTC  \n"
                f"- **Lower:** `{bot2:.6f}` BTC  \n"
                f"- **Levels:** `{opt2}`  \n"
                f"- **Step:** `{step2:.8f}`  \n"
                f"- **TP:** `{tp2:.6f}` BTC  \n"
                f"- **SL:** `{sl_amt:.6f}` BTC  \n"
                f"- **Per Order:** `{per_btc2:.6f}` BTC (`${per_usd2:.2f}`) "
                f"{'✅' if per_btc2>=MIN_ORDER else '❌'}"
            )
            with st.expander("📋 Copyable Summary"):
                summ2 = (
                    f"Upper: {top:.6f} BTC\nLower: {bot2:.6f} BTC\n"
                    f"Levels: {opt2}\nTP: {tp2:.6f} BTC\n"
                    f"SL: {sl_amt:.6f} BTC\n"
                    f"Per Order: {per_btc2:.6f} BTC (${per_usd2:.2f})"
                )
                st.code(summ2, language="text")
        else:
            st.info("No grid recommended at this time.")

# ── About ──
with st.expander("ℹ️ About"):
    st.markdown("""
    • Free assistant—no auto-trades.  
    • **BTC/USDT**: trend+volatility filters, TP & SL.  
    • **XRP/BTC**: mean-reversion with dynamic TP & SL, min-gap filter.  
    • Enter USD budget; shows BTC & USD per-order.  
    • Auto-min-order sized for Crypto.com’s 30 levels.  
    """)
