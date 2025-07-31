import streamlit as st
import requests, pandas as pd, numpy as np
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# ── Auto-refresh ──
st_autorefresh(interval=60_000, key="datarefresh")
st.set_page_config(page_title="Grid Bot Reset Assistant", layout="centered")

# ── Sidebar: Global Settings ──
st.sidebar.title("⚙️ Global Settings")
HISTORY_DAYS = st.sidebar.slider("History Window (days)", 30, 180, 90)
VOL_WINDOW   = st.sidebar.slider("Volatility Window (days)", 7, 30, 14)

# ── Sidebar: BTC Strategy Settings ──
st.sidebar.markdown("### 🟡 BTC/USDT Strategy")
RSI_OVER     = st.sidebar.slider("RSI Overbought Thresh", 60, 90, 75)
TP_MULT      = st.sidebar.slider("Take-Profit Multiplier", 0.5, 2.0, 1.0, 0.1,
                                  help="Multiply the 'drop' to set your TP above entry.")
STOP_LOSS_PCT= st.sidebar.slider("Stop-Loss % below bottom", 0.5, 5.0, 2.0, 0.5)
GRID_MIN, GRID_MAX = 1, 30

# ── Sidebar: XRP Strategy Settings ──
st.sidebar.markdown("### 🟣 XRP/BTC Strategy")
XRP_MEAN_D   = st.sidebar.slider("Mean Window (days)", 5, 20, 10)
XRP_SIGNAL_CD= st.sidebar.slider("Cooldown Between Resets (days)", 1, 7, 3)
XRP_MEAN_TGT = st.sidebar.slider("Bounce Target (%)", 50, 100, 100,
                                  help="What % of move back to mean to take profit.")

# ── Sidebar: Investment Settings ──
st.sidebar.markdown("### 💰 Investment Settings")
INV_BTC   = st.sidebar.number_input("Total Investment (BTC)", 1e-5, 0.1, 0.01, 1e-5)
MIN_ORDER = st.sidebar.number_input("Min Order Size (BTC)", 1e-6, 1e-3, 5e-4, 1e-6)

# ── Utility ──
def fetch_json(url, params):
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def get_btc_history():
    data = fetch_json("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
                      {"vs_currency":"usd","days":HISTORY_DAYS})["prices"]
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
    return df.dropna()

@st.cache_data(ttl=600)
def get_xrp_history():
    # Simulated daily XRP/BTC: replace with real API if desired
    np.random.seed(42)
    base = 0.02 + np.cumsum(np.random.normal(0,0.0015,HISTORY_DAYS))
    df = pd.DataFrame({"price":base}, index=pd.date_range(end=pd.Timestamp.today(),periods=HISTORY_DAYS))
    df["return"] = df["price"].pct_change()*100
    df["vol"]    = df["return"].rolling(VOL_WINDOW).std()
    df["mean"]   = df["price"].rolling(XRP_MEAN_D).mean()
    df["signal"] = (df["price"]<df["mean"]) & (df["vol"]>df["vol"].shift(1))
    return df.dropna()

@st.cache_data(ttl=60)
def get_live():
    j = fetch_json("https://api.coingecko.com/api/v3/simple/price",
                   {"ids":"bitcoin,ripple","vs_currencies":"usd,btc","include_24hr_change":"true"})
    return {
      "BTC":(j["bitcoin"]["usd"], j["bitcoin"].get("usd_24h_change",0)),
      "XRP":(j["ripple"]["btc"],  None)
    }

def compute_grid(top, drop, levels):
    bot = top*(1-drop/100)
    step= (top-bot)/levels
    return bot, step

# ── Fetch Data ──
btc_hist = get_btc_history()
xrp_hist = get_xrp_history()
live     = get_live()
btc_p, btc_ch = live["BTC"]
xrp_p, _      = live["XRP"]

now = datetime.now(pytz.timezone("Europe/London"))
st.title("🔁 Grid Bot Reset Assistant")
st.caption(f"Last updated: {now:%Y-%m-%d %H:%M %Z}")

tabs = st.tabs(["🟡 BTC/USDT", "🟣 XRP/BTC"])

with tabs[0]:
    st.header("BTC/USDT Strategy")
    LATEST = btc_hist.iloc[-1]
    mod, strg = LATEST["vol"], 2*LATEST["vol"]
    st.markdown(f"- **Price:** ${btc_p:.2f}  \n"
                f"- **24h Δ:** {btc_ch:.2f}%  \n"
                f"- **14d Vol:** {mod:.2f}%  \n"
                f"- **RSI:** {LATEST['rsi']:.1f}")
    # backtest for optimizing levels
    conds = ((btc_hist["return"]>=mod)&
             (btc_hist["price"]>btc_hist["ema50"])&
             (btc_hist["sma5"]>btc_hist["sma20"])&
             (btc_hist["rsi"]<RSI_OVER))
    trades = conds.sum()
    wins   = ((btc_hist["price"].shift(-1)>btc_hist["price"]) & conds).sum()
    wr = wins/trades if trades else 0

    # live decision
    drop=None; status="No reset"
    if btc_ch>=mod:
        drop = mod if btc_ch<=strg else strg
        status = "Moderate reset" if btc_ch<=strg else "Strong reset"
    st.markdown(f"- **Signal:** {status} (drop {drop:.2f}%)" if drop else "- **Signal:** None")

    ok = (LATEST["price"]>LATEST["ema50"] and LATEST["sma5"]>LATEST["sma20"] and LATEST["rsi"]<RSI_OVER)
    st.markdown(f"- **Filters OK:** {'✅' if ok else '❌'}")

    if drop and ok:
        # optimize L
        scores=[wr*(drop/L) for L in range(GRID_MIN,GRID_MAX+1)]
        opt = int(np.argmax(scores))+GRID_MIN
        bot, step = compute_grid(btc_p, drop, opt)
        tp = btc_p - bot
        tp_price = bot + tp*TP_MULT
        sl_price = bot*(1-STOP_LOSS_PCT/100)
        per = INV_BTC/opt
        valid= per>=MIN_ORDER

        st.subheader("📐 Grid Recommendation")
        st.markdown(f"- Upper: `{btc_p:.2f}`  \n"
                    f"- Lower: `{bot:.2f}`  \n"
                    f"- Levels: `{opt}`  \n"
                    f"- Step: `{step:.4f}`  \n"
                    f"- Take-Profit @ {TP_MULT:.1f}×drop → `{tp_price:.2f}`  \n"
                    f"- Stop-Loss @ {STOP_LOSS_PCT:.1f}% below → `{sl_price:.2f}`  \n"
                    f"- Per-Order: `{per:.6f}` BTC {'✅' if valid else '❌'}")

        with st.expander("📋 Copyable Summary"):
            summ = (f"Upper: {btc_p:.2f}\nLower: {bot:.2f}\nLevels: {opt}"
                    f"\nTP: {tp_price:.2f}\nSL: {sl_price:.2f}\nPer order: {per:.6f} BTC")
            st.code(summ, language="text")

    else:
        st.info("No grid recommended.")

    with st.expander("📊 Backtest Stats"):
        st.write(f"- Trades: {trades}, Wins: {wins}, Win Rate: {wr*100:.1f}%")

with tabs[1]:
    st.header("XRP/BTC Strategy")
    L = xrp_hist.iloc[-1]
    st.markdown(f"- **Price:** {xrp_p:.6f} BTC  \n"
                f"- **Mean({XRP_MEAN_D}d):** {L['mean']:.6f}  \n"
                f"- **Vol({VOL_WINDOW}d):** {L['vol']:.2f}%")
    sig = L["signal"]
    st.markdown(f"- **Signal:** {'✅ Reset' if sig else '❌ None'}")

    if sig:
        # backtest win‐rate for mean reversion
        df=xrp_hist.copy()
        df["win"]=((df["price"].shift(-1)>df["mean"]) & df["signal"])
        wr2 = df["win"].sum()/df["signal"].sum()
        # optimize levels
        scores=[wr2*(L["vol"]/LL) for LL in range(GRID_MIN,GRID_MAX+1)]
        opt2=int(np.argmax(scores))+GRID_MIN
        top=L["mean"]; bot= xrp_p*(1-L["vol"]/100)
        step2=(top-bot)/opt2
        bounce=(top-xrp_p)*XRP_MEAN_TGT/100
        tp2= xrp_p + bounce
        per2=INV_BTC/opt2
        ok2= per2>=MIN_ORDER

        st.subheader("📐 Grid Recommendation")
        st.markdown(f"- Upper: `{top:.6f}`  \n"
                    f"- Lower: `{bot:.6f}`  \n"
                    f"- Levels: `{opt2}`  \n"
                    f"- Step: `{step2:.8f}`  \n"
                    f"- Bounce-TP @ {XRP_MEAN_TGT}% → `{tp2:.6f}`  \n"
                    f"- Per-Order: `{per2:.6f}` BTC {'✅' if ok2 else '❌'}")

        with st.expander("📋 Copyable Summary"):
            summ2=(f"Upper: {top:.6f}\nLower: {bot:.6f}\nLevels: {opt2}"
                   f"\nTP: {tp2:.6f}\nPer order: {per2:.6f} BTC")
            st.code(summ2, language="text")

        with st.expander("📊 Signal History"):
            total=xrp_hist["signal"].sum()
            st.write(f"- Signals last {HISTORY_DAYS} d: {total}, Win Rate: {wr2*100:.1f}%")
    else:
        st.info("No grid recommended.")

# ── Disclaimer ──
with st.expander("ℹ️ About"):
    st.markdown("""
    • Free manual assistant — does not place orders.  
    • Copy parameters into your Crypto.com Grid Bot.  
    • All data via CoinGecko (free tier).  
    • Use responsibly; backtests are illustrative.
    """)
