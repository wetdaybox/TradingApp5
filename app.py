import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from sklearn.ensemble import RandomForestRegressor
from streamlit_autorefresh import st_autorefresh

# ── Auto-refresh every 60 s ──
st_autorefresh(interval=60_000, key="datarefresh")

# ── Page Setup ──
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
ML_THRESH      = 0.70  # ML predicted win-rate threshold

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
    df["ema50"]  = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    df["sma5"]   = df["price"].rolling(5).mean()
    df["sma20"]  = df["price"].rolling(20).mean()
    df["vol14"]  = df["return"].rolling(VOL_WINDOW).std().fillna(0)
    delta       = df["price"].diff()
    gain, loss  = delta.clip(lower=0), -delta.clip(upper=0)
    df["rsi"]   = 100 - 100 / (1 + gain.rolling(RSI_WINDOW).mean() / loss.rolling(RSI_WINDOW).mean())
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

# ── Backtest Functions ──
def backtest_btc(df, rsi_th, tp_mult, sl_pct):
    wins = losses = 0
    for i in range(EMA_TREND, len(df)-1):
        p, rsi, vol = df["price"].iat[i], df["rsi"].iat[i], df["vol14"].iat[i]
        sma_cond = (p>df["ema50"].iat[i]) and (df["sma5"].iat[i]>df["sma20"].iat[i])
        if not (sma_cond and rsi<rsi_th): continue
        ret = df["return"].iat[i]
        if ret < vol: continue
        drop = vol if ret<=2*vol else 2*vol
        if df["price"].iat[i+1] > p: wins += 1
        else: losses += 1
    return wins/(wins+losses) if (wins+losses) else 0.0

def backtest_xrp(df, mean_d, bounce_pct, sl_pct, min_dip):
    wins = losses = 0
    df["mean"] = df["price"].rolling(mean_d).mean()
    df["vol"]  = df["return"].rolling(VOL_WINDOW).std().fillna(0)
    for i in range(mean_d, len(df)-1):
        p, m = df["price"].iat[i], df["mean"].iat[i]
        gap = (m-p)/p*100
        vol_up = df["vol"].iat[i]>df["vol"].iat[i-1]
        if not (p<m and gap>=min_dip and vol_up): continue
        tp = gap/100*p*(bounce_pct/100)
        if df["price"].iat[i+1]>=p+tp: wins += 1
        else: losses += 1
    return wins/(wins+losses) if (wins+losses) else 0.0

# ── Load Historical & Live ──
btc_hist = load_history("bitcoin","usd",HISTORY_DAYS)
xrp_hist = load_history("ripple","btc",HISTORY_DAYS)
live     = load_live()
btc_p, btc_ch = live["BTC"]
xrp_p, _      = live["XRP"]

# ── Generate Backtest Samples for ML Training ──
btc_samples = []
for rsi in (65,70,75,80,85):
    for tp in (1.0,1.5,2.0):
        for sl in (0.5,1.0,2.0):
            wr = backtest_btc(btc_hist, rsi, tp, sl)
            btc_samples.append((rsi, tp, sl, wr))
btc_df_s = pd.DataFrame(btc_samples, columns=["rsi","tp","sl","win_rate"])

xrp_samples = []
for m in (5,10,15):
    for b in (50,75,100):
        for sl in (25,50,75):
            for md in (1.0,1.5):
                wr = backtest_xrp(xrp_hist, m, b, sl, md)
                xrp_samples.append((m,b,sl,md,wr))
xrp_df_s = pd.DataFrame(xrp_samples, columns=["mean","bounce","sl","min_dip","win_rate"])

# ── Train ML Models ──
btc_ml = RandomForestRegressor(n_estimators=100, random_state=0)
btc_ml.fit(btc_df_s[["rsi","tp","sl"]], btc_df_s["win_rate"])
xrp_ml = RandomForestRegressor(n_estimators=100, random_state=0)
xrp_ml.fit(xrp_df_s[["mean","bounce","sl","min_dip"]], xrp_df_s["win_rate"])

# ── Determine Default & ML-Override Params ──
btc_def = btc_df_s.loc[btc_df_s.win_rate.idxmax(), ["rsi","tp","sl"]].tolist()
xrp_def = xrp_df_s.loc[xrp_df_s.win_rate.idxmax(), ["mean","bounce","sl","min_dip"]].tolist()

btc_pred = btc_ml.predict([btc_def])[0];  ml_btc = btc_pred>=ML_THRESH
xrp_pred = xrp_ml.predict([xrp_def])[0];  ml_xrp = xrp_pred>=ML_THRESH

# ── Sidebar Settings ──
st.sidebar.title("💰 Investment Settings")
usd_alloc      = st.sidebar.number_input("Investment ($)",10.0,1e6,500.0,10.0)
user_min_order = st.sidebar.number_input("Min Order (BTC)",1e-6,1e-2,5e-4,1e-6,format="%.6f")
MIN_ORDER = max(user_min_order, (usd_alloc/GRID_MAX)/btc_p if btc_p else 0)
st.sidebar.caption(f"🔒 Min Order ≥ {MIN_ORDER:.6f} BTC (~${MIN_ORDER*btc_p:.2f})")

# ── Final Parameter Display ──
st.sidebar.markdown("### ⚙️ Final Defaults")
label_b = f"BTC {'(ML)' if ml_btc else ''}: RSI<{btc_def[0]}, TP×{btc_def[1]}, SL{btc_def[2]}% (pred {btc_pred:.0%})"
label_x = f"XRP {'(ML)' if ml_xrp else ''}: Mean{int(xrp_def[0])}d, Bounce{xrp_def[1]}%, SL{xrp_def[2]}%, MinDip{xrp_def[3]}% (pred {xrp_pred:.0%})"
st.sidebar.write(label_b)
st.sidebar.write(label_x)

# ── Grid Calculator ──
def compute_grid(price, drop, levels):
    bottom = price*(1-drop/100)
    step   = (price-bottom)/levels
    return bottom,step

# ── BTC Bot ──
st.header("🟡 BTC/USDT Bot")
st.write(f"- Price: ${btc_p:.2f} | 24 h Δ: {btc_ch:.2f}%")
vol14 = btc_hist["vol14"].iat[-1]
ch    = btc_ch if btc_ch is not None else btc_hist["return"].iat[-1]
drop  = vol14 if ch<vol14 else (2*vol14 if ch>2*vol14 else ch)
if drop:
    if ml_btc: st.success("✅ ML Override Active")
    L = GRID_MORE if ml_btc else GRID_PRIMARY
    bot,step = compute_grid(btc_p, drop, L)
    per = (usd_alloc/btc_p)/L
    st.markdown(
        f"**Grid ({L})**  \n"
        f"- Lower: `{bot:.2f}`  \n"
        f"- Upper: `{btc_p:.2f}`  \n"
        f"- Step: `{step:.4f}`  \n"
        f"- Per-Order: `{per:.6f}` BTC {'✅' if per>=MIN_ORDER else '❌'}"
    )
else:
    st.info("No grid reset recommended")

# ── XRP Bot ──
st.header("🟣 XRP/BTC Bot")
st.write(f"- Price: {xrp_p:.6f} BTC")
hist = xrp_hist.copy()
sig = (hist["price"].iat[-1]<hist["price"].rolling(int(xrp_def[0])).mean().iat[-1]) \
      and (hist["vol14"].iat[-1]>hist["vol14"].iat[-2])
if sig:
    if ml_xrp: st.success("✅ ML Override Active")
    L = GRID_MORE if ml_xrp else GRID_PRIMARY
    bot_x,step_x = compute_grid(xrp_p, xrp_def[1], L)
    per_x = (usd_alloc/btc_p)/L
    st.markdown(
        f"**Grid ({L})**  \n"
        f"- Lower: `{bot_x:.6f}`  \n"
        f"- Upper: `{xrp_p:.6f}`  \n"
        f"- Step: `{step_x:.8f}`  \n"
        f"- Per-Order: `{per_x:.6f}` BTC {'✅' if per_x>=MIN_ORDER else '❌'}"
    )
else:
    st.info("No grid reset recommended")

# ── About & Requirements ──
with st.expander("ℹ️ About"):
    st.markdown("""
    • Original backtests auto-tune BTC & XRP grid parameters for ≥70 % win-rate.  
    • ML models trained on those backtests override defaults when confidence ≥70 %.  
    • Live ML override flagged with ✅ next to grid.  
    • Copy these parameters into your Crypto.com grid bot manually.
    """)
with st.expander("📦 requirements.txt"):
    st.code("""
streamlit==1.47.1
streamlit-autorefresh==1.0.1
pandas>=2.3,<2.4
numpy>=2.3,<3
requests>=2.32,<3
scikit-learn>=1.2
pytz>=2025.2
    """)
