import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from streamlit_autorefresh import st_autorefresh
import time

# ── Auto-refresh every 60 s ──
st_autorefresh(interval=60_000, key="datarefresh")

# ── Page Setup ──
st.set_page_config(layout="centered")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
now = datetime.now(pytz.timezone("Europe/London"))
st.caption(f"Last updated: {now:%Y-%m-%d %H:%M %Z}")

# ── Constants ──
HISTORY_DAYS      = 90
VOL_WINDOW        = 14
RSI_WINDOW        = 14
EMA_TREND         = 50
GRID_PRIMARY      = 20
GRID_MAX          = 30
CLASS_PROB_THRESH = 0.80
MAX_RETRIES       = 3

# ── Helpers ──
def fetch_json(url, params):
    for i in range(MAX_RETRIES):
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 429:
            time.sleep(2**i)
            continue
        r.raise_for_status()
        return r.json()
    st.warning("⚠️ Rate limit reached; using cached data.")
    return {}

@st.cache_data(ttl=600)
def load_history(coin_id, vs, days):
    js     = fetch_json(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
                {"vs_currency": vs, "days": days}
             ) or {}
    prices = js.get("prices", [])
    df     = pd.DataFrame(prices, columns=["ts","price"])
    if df.empty:
        return df
    df["date"]   = pd.to_datetime(df["ts"], unit="ms")
    df           = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change() * 100
    df["ema50"]  = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    df["sma5"]   = df["price"].rolling(5).mean()
    df["sma20"]  = df["price"].rolling(20).mean()
    df["vol14"]  = df["return"].rolling(VOL_WINDOW).std().fillna(0)
    delta        = df["price"].diff()
    gain         = delta.clip(lower=0)
    loss         = -delta.clip(upper=0)
    df["rsi"]    = 100 - 100 / (
                       1 + gain.rolling(RSI_WINDOW).mean()/
                           loss.rolling(RSI_WINDOW).mean()
                   )
    return df.dropna()

@st.cache_data(ttl=60)
def load_live():
    js  = fetch_json(
            "https://api.coingecko.com/api/v3/simple/price",
            {"ids":"bitcoin,ripple","vs_currencies":"usd,btc","include_24hr_change":"true"}
          ) or {}
    btc = js.get("bitcoin", {})
    xrp = js.get("ripple", {})
    return {
        "BTC": (btc.get("usd", np.nan), btc.get("usd_24h_change", np.nan)),
        "XRP": (xrp.get("btc", np.nan), None)
    }

def generate_signals(df, is_btc, params):
    X, y = [], []
    for i in range(EMA_TREND, len(df)-1):
        p        = df["price"].iat[i]
        ret      = df["return"].iat[i]
        vol      = df["vol14"].iat[i]
        ema_diff = p - df["ema50"].iat[i]
        mom      = df["sma5"].iat[i] - df["sma20"].iat[i]
        rsi      = df["rsi"].iat[i]

        if is_btc:
            rsi_th, tp, sl = params
            cond = (ema_diff>0) and (mom>0) and (rsi<rsi_th) and (ret>=vol)
        else:
            mean_d, bounce, sl, dip = params
            mp   = df["price"].rolling(mean_d).mean().iat[i]
            cond = (p<mp) and (((mp-p)/p*100)>=dip) and (vol>df["vol14"].iat[i-1])

        if not cond:
            continue

        X.append([rsi, vol, ema_diff, mom, ret])
        profit = df["price"].iat[i+1] - p
        y.append(1 if profit>0 else 0)

    return np.array(X), np.array(y)

# ── Load Data ──
btc_hist    = load_history("bitcoin","usd",HISTORY_DAYS)
xrp_hist    = load_history("ripple","btc",HISTORY_DAYS)
btc_p,btc_ch = load_live()["BTC"]
xrp_p,_      = load_live()["XRP"]

# ── Default Parameters ──
btc_params = (75, 1.5, 1.0)
xrp_params = (10, 75, 50, 1.0)

# ── Build Training Data ──
X_btc, y_btc = generate_signals(btc_hist, True,  btc_params)
X_xrp, y_xrp = generate_signals(xrp_hist, False, xrp_params)

# ── Train with fallback ──
def train_clf(X, y):
    if len(y)>=6 and len(np.unique(y))>1:
        gs = GridSearchCV(
            RandomForestClassifier(random_state=0),
            {"n_estimators":[50,100],"max_depth":[3,5]},
            cv=3, scoring="accuracy", n_jobs=-1
        )
        gs.fit(X, y)
        return gs.best_estimator_
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    if len(y)>0:
        clf.fit(X, y)
    return clf

clf_btc = train_clf(X_btc, y_btc)
clf_xrp = train_clf(X_xrp, y_xrp)

# ── Today’s Features (guard empty) ──
def today_feat(df):
    if df is None or df.empty:
        return None
    i = len(df)-1
    return [[
        df["rsi"].iat[i],
        df["vol14"].iat[i],
        df["price"].iat[i] - df["ema50"].iat[i],
        df["sma5"].iat[i] - df["sma20"].iat[i],
        df["return"].iat[i],
    ]]

# ── Safe proba ──
def safe_prob(clf, feat):
    if feat is None:
        return 0.0
    probs = clf.predict_proba(feat)[0]
    return probs[1] if probs.shape[0]>1 else 0.0

p_btc = safe_prob(clf_btc, today_feat(btc_hist))
p_xrp = safe_prob(clf_xrp, today_feat(xrp_hist))

use_ml_btc = p_btc >= CLASS_PROB_THRESH
use_ml_xrp = p_xrp >= CLASS_PROB_THRESH

# ── Sidebar: Investment & Split ──
st.sidebar.title("💰 Investment Settings")
usd_total      = st.sidebar.number_input("Total Investment ($)",100.0,1e6,3000.0,100.0)
pct_btc        = st.sidebar.slider("BTC Allocation (%)",0,100,70)
usd_btc_alloc  = usd_total * pct_btc/100
usd_xrp_alloc  = usd_total - usd_btc_alloc
gbp_rate       = st.sidebar.number_input("GBP/USD Rate",1.1,1.6,1.27,0.01)
st.sidebar.metric("Total Value (USD/GBP)",f"${usd_total:,.2f}",f"£{usd_total/gbp_rate:,.2f}")
min_order      = st.sidebar.number_input("Min Order (BTC)",1e-6,1e-2,5e-4,1e-6,format="%.6f")
MIN_ORDER      = max(min_order,(usd_btc_alloc/GRID_MAX)/btc_p if btc_p else 0)
st.sidebar.caption(f"🔒 Min Order ≥ {MIN_ORDER:.6f} BTC (~${MIN_ORDER*btc_p:.2f})")

# ── Final Defaults ──
st.sidebar.markdown("### ⚙️ Final Defaults")
st.sidebar.write(f"BTC{' (ML)' if use_ml_btc else ''}: RSI<{btc_params[0]},TP×{btc_params[1]},SL{btc_params[2]}% (p={p_btc:.0%})")
st.sidebar.write(f"XRP{' (ML)' if use_ml_xrp else ''}: Mean{xrp_params[0]}d,Bounce{xrp_params[1]}%,SL{xrp_params[2]}%,Dip{xrp_params[3]}% (p={p_xrp:.0%})")

# ── Display Logic ──
def display_bot(name, price, drop, levels, ml_flag, usd_alloc, is_btc):
    st.header(name)
    unit = "USD" if is_btc else "BTC"
    st.write(f"- Price: {price:.6f} {unit}")
    if drop>0:
        if ml_flag:
            st.success("✅ ML Override Active")
        st.write(f"- Grid Depth (Drop %): {drop:.2f}%")
        st.write("_How far below current price the lowest grid level sits._")
        low  = price*(1-drop/100)
        step = (price-low)/levels
        per  = (usd_alloc/price)/levels if is_btc else (usd_alloc/btc_p)/levels
        st.write(f"- Lower: {low:.6f}; Upper: {price:.6f}; Step: {step:.6f}")
        st.write(f"- Per-Order: {per:.6f} BTC {'✅' if per>=MIN_ORDER else '❌'}")
        tp = price*(1+drop/100) if is_btc else price*(1+xrp_params[1]/100)
        st.write(f"- Take-Profit: {tp:.6f} {unit}")
        st.write("🔄 Redeploy Bot now")
    else:
        st.error("🛑 Terminate Bot")

vol14     = btc_hist["vol14"].iat[-1] if not btc_hist.empty else 0
ret24     = btc_ch if btc_ch is not None else (btc_hist["return"].iat[-1] if not btc_hist.empty else 0)
drop_btc  = vol14 if ret24<vol14 else (2*vol14 if ret24>2*vol14 else ret24)
levels_b  = GRID_PRIMARY if not use_ml_btc else GRID_MAX
display_bot("🟡 BTC/USDT Bot", btc_p, drop_btc, levels_b, use_ml_btc, usd_btc_alloc, True)

sig_xrp   = (not xrp_hist.empty) and \
            (xrp_hist["price"].iat[-1] < xrp_hist["price"].rolling(int(xrp_params[0])).mean().iat[-1]) and \
            (xrp_hist["vol14"].iat[-1] > xrp_hist["vol14"].iat[-2])
drop_xrp  = xrp_params[1] if sig_xrp else 0
levels_x  = GRID_PRIMARY if not use_ml_xrp else GRID_MAX
display_bot("🟣 XRP/BTC Bot", xrp_p, drop_xrp, levels_x, use_ml_xrp, usd_xrp_alloc, False)

# ── About & Requirements ──
with st.expander("ℹ️ About"):
    st.markdown("""
    1. **Set your total investment and BTC allocation** in the sidebar.
    2. **Min Order** ensures grid orders meet your exchange’s minimum.
    3. **Final Defaults** shows rule-based or ML (p(win)≥80%) parameters.
    4. **Grid Depth (Drop %)** is the percentage below current price to place
       the bottom of your grid—_not_ a past drop.
    5. **Per-Order (BTC)** is your BTC allocation per grid line (green if above min).
    6. **Take-Profit** is the target price to exit profitably.
    7. **🔄 Redeploy Bot now** when grid is recommended; **🛑 Terminate Bot** if no signal.
    8. Page auto-refreshes every 60 s with new live data.
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
