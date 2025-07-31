import streamlit as st
import requests, pandas as pd, numpy as np
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# ── Auto-refresh ──
st_autorefresh(interval=60_000, key="datarefresh")
st.set_page_config(page_title="Infinite Scalping Grid Bot Trading System", layout="centered")

# ── Sidebar Settings ──
st.sidebar.title("⚙️ Settings")
HISTORY_DAYS  = st.sidebar.slider("History Window (days)", 30, 180, 90)
VOL_WINDOW    = st.sidebar.slider("Volatility Window (days)", 7, 30, 14)

st.sidebar.markdown("### 🟡 BTC/USDT Strategy")
RSI_OVER      = st.sidebar.slider("RSI Overbought Threshold", 60, 90, 75)
TP_MULT       = st.sidebar.slider("Take-Profit Multiplier", 0.5, 2.0, 1.0, 0.1)
STOP_LOSS_PCT = st.sidebar.slider("Stop-Loss % below bottom", 0.5, 5.0, 2.0, 0.5)

st.sidebar.markdown("### 🟣 XRP/BTC Strategy")
XRP_MEAN_D    = st.sidebar.slider("Mean Window (days)", 5, 20, 10)
XRP_TGT       = st.sidebar.slider("Bounce Target (%)", 50, 100, 100)

st.sidebar.markdown("### 💰 Investment")
INV_BTC       = st.sidebar.number_input("Total Investment (BTC)", 1e-5, 0.1, 0.01, 1e-5)
MIN_ORDER     = st.sidebar.number_input("Min Order Size (BTC)", 1e-6, 1e-3, 5e-4, 1e-6)

GRID_MIN, GRID_MAX = 1, 30

# ── Fallback caches ──
_last_btc_df = None
_last_live   = None

# ── Helpers ──
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
        st.warning("Rate-limited on BTC history; using last cached data.")
        return _last_btc_df if _last_btc_df is not None else pd.DataFrame()

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
        st.warning("Rate-limited on live price; using last cached data.")
        return _last_live if _last_live is not None else {"BTC":(None,None),"XRP":(None,None)}

def compute_grid(top, drop, levels):
    bottom = top * (1 - drop/100)
    step   = (top - bottom) / levels
    return bottom, step

# ── Load Data ──
btc_hist    = get_btc_history()
live_prices = get_live()
btc_p, btc_ch = live_prices["BTC"]

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
xrp_p, _ = live_prices["XRP"]

# ── Header ──
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
now = datetime.now(pytz.timezone("Europe/London"))
st.caption(f"Last updated: {now:%Y-%m-%d %H:%M %Z}")

# ── Tabs ──
tab1, tab2 = st.tabs(["🟡 BTC/USDT", "🟣 XRP/BTC"])

# ── BTC/USDT ──
with tab1:
    if btc_hist.empty or btc_p is None:
        st.error("BTC data unavailable.")
    else:
        L = btc_hist.iloc[-1]
        mod, strg = L["vol"], 2*L["vol"]
        st.markdown(
            f"- **Price:** ${btc_p:.2f}  \n"
            f"- **24h Δ:** {btc_ch:.2f}%  \n"
            f"- **Vol(14d):** {mod:.2f}%  \n"
            f"- **RSI:** {L['rsi']:.1f}"
        )
        drop, status = None, "No grid recommended"
        if btc_ch >= mod:
            drop = mod if btc_ch <= strg else strg
            status = "Moderate reset" if btc_ch <= strg else "Strong reset"
        st.markdown(f"- **Signal:** {status}")
        ok = (L["price"]>L["ema50"] and L["sma5"]>L["sma20"] and L["rsi"]<RSI_OVER)
        st.markdown(f"- **Filters OK:** {'✅' if ok else '❌'}")

        if drop and ok:
            cond   = ((btc_hist["return"]>=mod)&
                      (btc_hist["price"]>btc_hist["ema50"])&
                      (btc_hist["sma5"]>btc_hist["sma20"])&
                      (btc_hist["rsi"]<RSI_OVER))
            trades = cond.sum()
            wins   = ((btc_hist["price"].shift(-1)>btc_hist["price"])&cond).sum()
            wr     = wins/trades if trades else 0

            scores = [wr*(drop/L) for L in range(GRID_MIN,GRID_MAX+1)]
            opt    = int(np.argmax(scores))+GRID_MIN
            bot, step = compute_grid(btc_p, drop, opt)
            tp_price  = bot + (btc_p-bot)*TP_MULT
            sl_price  = bot*(1-STOP_LOSS_PCT/100)
            per       = INV_BTC/opt

            st.subheader("📐 Grid Recommendation")
            st.markdown(
                f"- Upper: `{btc_p:.2f}`  \n"
                f"- Lower: `{bot:.2f}`  \n"
                f"- Levels: `{opt}`  \n"
                f"- Step: `{step:.4f}`  \n"
                f"- TP: `{tp_price:.2f}`  \n"
                f"- SL: `{sl_price:.2f}`  \n"
                f"- Per-Order: `{per:.6f}` BTC {'✅' if per>=MIN_ORDER else '❌'}"
            )
            with st.expander("📋 Copyable Summary"):
                summ = (
                    f"Upper: {btc_p:.2f}\nLower: {bot:.2f}\n"
                    f"Levels: {opt}\nTP: {tp_price:.2f}\n"
                    f"SL: {sl_price:.2f}\nPer order: {per:.6f} BTC"
                )
                st.code(summ, language="text")
        else:
            st.info("No grid recommended at this time.")

# ── XRP/BTC ──
with tab2:
    if xrp_hist.empty or xrp_p is None:
        st.error("XRP data unavailable.")
    else:
        L = xrp_hist.iloc[-1]
        st.markdown(
            f"- **Price:** {xrp_p:.6f} BTC  \n"
            f"- **Mean({XRP_MEAN_D}d):** {L['mean']:.6f}  \n"
            f"- **Vol({VOL_WINDOW}d):** {L['vol']:.2f}%"
        )
        sig = L["signal"]
        st.markdown(f"- **Signal:** {'✅ Reset' if sig else '❌ None'}")

        if sig:
            df2 = xrp_hist.copy()
            df2["win"] = ((df2["price"].shift(-1)>df2["mean"]) & df2["signal"])
            wr2 = df2["win"].sum()/df2["signal"].sum() if df2["signal"].sum() else 0

            scores2 = [wr2*(L["vol"]/lvl) for lvl in range(GRID_MIN,GRID_MAX+1)]
            opt2    = int(np.argmax(scores2))+GRID_MIN
            top     = L["mean"]
            bot     = xrp_p*(1-L["vol"]/100)
            step2   = (top-bot)/opt2
            bounce  = (top-xrp_p)*(XRP_TGT/100)
            tp2     = xrp_p + bounce
            per2    = INV_BTC/opt2

            st.subheader("📐 Grid Recommendation")
            st.markdown(
                f"- Upper: `{top:.6f}` BTC  \n"
                f"- Lower: `{bot:.6f}` BTC  \n"
                f"- Levels: `{opt2}`  \n"
                f"- Step: `{step2:.8f}`  \n"
                f"- TP: `{tp2:.6f}` BTC  \n"
                f"- Per-Order: `{per2:.6f}` BTC {'✅' if per2>=MIN_ORDER else '❌'}"
            )
            with st.expander("📋 Copyable Summary"):
                summ2 = (
                    f"Upper: {top:.6f} BTC\nLower: {bot:.6f} BTC\n"
                    f"Levels: {opt2}\nTP: {tp2:.6f} BTC\n"
                    f"Per order: {per2:.6f} BTC"
                )
                st.code(summ2, language="text")
        else:
            st.info("No grid recommended at this time.")

# ── Disclaimer ──
with st.expander("ℹ️ About"):
    st.markdown("""
    • Free manual assistant — does not place orders.  
    • CoinGecko (free tier) with graceful 429 fallback.  
    • **BTC/USDT**: trend, volatility, RSI, TP & SL.  
    • **XRP/BTC**: mean-reversion with optimized levels.  
    • Copy the “Copyable Summary” into your Crypto.com Grid Bot.  
    """)
