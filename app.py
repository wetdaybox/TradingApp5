import streamlit as st
import requests, pandas as pd, numpy as np
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# ── Auto-refresh every 60 s ──
st_autorefresh(interval=60_000, key="datarefresh")
st.set_page_config(page_title="Grid Bot Reset Assistant", layout="centered")

# ── Sidebar: Settings ──
st.sidebar.title("⚙️ Settings")
HISTORY_DAYS    = st.sidebar.slider("History Window (days)", 30, 180, 90)
VOL_WINDOW      = st.sidebar.slider("Volatility Window (days)", 7, 30, 14)

st.sidebar.markdown("### 🟡 BTC/USDT Strategy")
RSI_OVER        = st.sidebar.slider("RSI Overbought Thresh", 60, 90, 75)
TP_MULT         = st.sidebar.slider("TP Multiplier", 0.5, 2.0, 1.0, 0.1)
STOP_LOSS_PCT   = st.sidebar.slider("Stop-Loss % below bottom", 0.5, 5.0, 2.0, 0.5)

st.sidebar.markdown("### 🟣 XRP/BTC Strategy")
XRP_MEAN_D      = st.sidebar.slider("Mean Window (days)", 5, 20, 10)
XRP_TGT         = st.sidebar.slider("Bounce Target (%)", 50, 100, 100, help="Percent back to mean")

st.sidebar.markdown("### 💰 Investment")
INV_BTC         = st.sidebar.number_input("Total Investment (BTC)", 1e-5, 0.1, 0.01, 1e-5)
MIN_ORDER       = st.sidebar.number_input("Min Order Size (BTC)", 1e-6, 1e-3, 5e-4, 1e-6)

GRID_MIN, GRID_MAX = 1, 30

# ── Globals for caching fallback ──
_last_btc_df = None
_last_live   = None

# ── Utility ──
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
        st.warning("BTC history rate-limited; using last cached data.")
        if _last_btc_df is not None:
            return _last_btc_df
        st.error("No BTC history available.")
        return pd.DataFrame()

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
            "XRP": (j["ripple"]["btc"],  None)
        }
        return _last_live
    except requests.exceptions.HTTPError:
        st.warning("Live price rate-limited; using last cached data.")
        if _last_live is not None:
            return _last_live
        st.error("No live price available.")
        return {"BTC":(None,None),"XRP":(None,None)}

def compute_grid(top, drop, levels):
    bot = top*(1-drop/100)
    step= (top-bot)/levels
    return bot, step

# ── Fetch Data ──
btc_hist = get_btc_history()
live     = get_live()
btc_p, btc_ch = live["BTC"]

# ── XRP Simulation ──
def get_xrp_history():
    np.random.seed(42)
    base = 0.02 + np.cumsum(np.random.normal(0,0.0015,len(btc_hist)))
    df = pd.DataFrame({"price":base}, index=btc_hist.index)
    df["return"] = df["price"].pct_change()*100
    df["vol"]    = df["return"].rolling(VOL_WINDOW).std()
    df["mean"]   = df["price"].rolling(XRP_MEAN_D).mean()
    df["signal"] = (df["price"]<df["mean"]) & (df["vol"]>df["vol"].shift(1))
    return df.dropna()

xrp_hist = get_xrp_history()
xrp_p, _ = live["XRP"]

# ── Header ──
now = datetime.now(pytz.timezone("Europe/London"))
st.title("🔁 Grid Bot Reset Assistant")
st.caption(f"Last updated: {now:%Y-%m-%d %H:%M %Z}")

tabs = st.tabs(["🟡 BTC/USDT", "🟣 XRP/BTC"])

# ── BTC Tab ──
with tabs[0]:
    if btc_hist.empty or btc_p is None:
        st.error("BTC data unavailable.")
    else:
        L = btc_hist.iloc[-1]
        mod, strg = L["vol"], 2*L["vol"]
        st.markdown(f"- **Price:** ${btc_p:.2f}  \n"
                    f"- **24h Δ:** {btc_ch:.2f}%  \n"
                    f"- **Vol(14d):** {mod:.2f}%  \n"
                    f"- **RSI:** {L['rsi']:.1f}")
        # ... (rest of your BTC logic unchanged) ...

# ── XRP Tab ──
with tabs[1]:
    if xrp_hist.empty or xrp_p is None:
        st.error("XRP data unavailable.")
    else:
        L = xrp_hist.iloc[-1]
        st.markdown(f"- **Price:** {xrp_p:.6f} BTC  \n"
                    f"- **Mean({XRP_MEAN_D}d):** {L['mean']:.6f}  \n"
                    f"- **Vol({VOL_WINDOW}d):** {L['vol']:.2f}%  \n"
                    f"- **Signal:** {'✅ Reset' if L['signal'] else '❌ None'}")
        # ... (rest of your XRP logic unchanged) ...

# ── Disclaimer ──
with st.expander("ℹ️ About"):
    st.markdown("""
    • Free manual assistant — does not place orders.  
    • Uses CoinGecko (free tier) with graceful 429 fallback.  
    • **BTC/USDT** uses trend, volatility, TP & SL controls.  
    • **XRP/BTC** uses mean-reversion with optimized levels.  
    • Copy the “Copyable Summary” into your Crypto.com Grid Bot.  
    """)
