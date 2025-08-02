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
HISTORY_DAYS       = 90
VOL_WINDOW         = 14
RSI_WINDOW         = 14
EMA_TREND          = 50
GRID_PRIMARY       = 20
GRID_MAX           = 30
CLASS_PROB_THRESH  = 0.80
MAX_RETRIES        = 3

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
    js = fetch_json(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        {"vs_currency": vs, "days": days}
    ) or {}
    prices = js.get("prices", [])
    df = pd.DataFrame(prices, columns=["ts","price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change()*100
    df["ema50"]  = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    df["sma5"]   = df["price"].rolling(5).mean()
    df["sma20"]  = df["price"].rolling(20).mean()
    df["vol14"]  = df["return"].rolling(VOL_WINDOW).std().fillna(0)
    d, g, l = df["price"].diff(), None, None
    df["rsi"]   = 100 - 100 / (
        1 + d.clip(lower=0).rolling(RSI_WINDOW).mean()/
        d.clip(upper=0).abs().rolling(RSI_WINDOW).mean()
    )
    return df.dropna()

@st.cache_data(ttl=60)
def load_live():
    js = fetch_json(
        "https://api.coingecko.com/api/v3/simple/price",
        {"ids":"bitcoin,ripple","vs_currencies":"usd,btc","include_24hr_change":"true"}
    ) or {}
    btc = js.get("bitcoin",{}); xrp = js.get("ripple",{})
    return {
        "BTC": (btc.get("usd", np.nan), btc.get("usd_24h_change", np.nan)),
        "XRP": (xrp.get("btc", np.nan), None)
    }

# ── Signal Generation & Labeling ──
def generate_signals(df, is_btc, params):
    X, y = [], []
    for i in range(EMA_TREND, len(df)-1):
        p   = df["price"].iat[i]
        ret = df["return"].iat[i]
        vol = df["vol14"].iat[i]
        ema_diff = p - df["ema50"].iat[i]
        mom      = df["sma5"].iat[i] - df["sma20"].iat[i]
        rsi      = df["rsi"].iat[i]

        if is_btc:
            rsi_th, tp, sl = params
            cond = (ema_diff>0) and (mom>0) and (rsi<rsi_th) and (ret>=vol)
        else:
            mean_d, bounce, sl, dip = params
            mp = df["price"].rolling(mean_d).mean().iat[i]
            cond = (p<mp) and (((mp-p)/p*100)>=dip) and (vol>df["vol14"].iat[i-1])

        if not cond:
            continue

        X.append([rsi, vol, ema_diff, mom, ret])
        profit = df["price"].iat[i+1] - p
        y.append(1 if profit>0 else 0)

    return np.array(X), np.array(y)

# ── Load Data ──
btc_hist = load_history("bitcoin","usd",HISTORY_DAYS)
xrp_hist = load_history("ripple","btc",HISTORY_DAYS)
btc_p, btc_ch = load_live()["BTC"]
xrp_p, _      = load_live()["XRP"]

# ── Defaults (from your backtest grid-search) ──
btc_params = (75, 1.5, 1.0)     # (RSI_th, TP×, SL%)
xrp_params = (10, 75, 50, 1.0)  # (mean_d, bounce%, SL%, dip%)

# ── Build Training Sets ──
X_btc, y_btc = generate_signals(btc_hist, True,  btc_params)
X_xrp, y_xrp = generate_signals(xrp_hist, False, xrp_params)

# ── Train or GridSearch (only if ≥3 samples) ──
def train_clf(X, y):
    if len(y) >= 3:
        grid = {"n_estimators":[50,100], "max_depth":[3,5]}
        gs = GridSearchCV(RandomForestClassifier(random_state=0),
                          grid, cv=3, scoring="accuracy", n_jobs=-1)
        gs.fit(X, y)
        return gs.best_estimator_
    # else fallback
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)
    return clf

clf_btc = train_clf(X_btc, y_btc)
clf_xrp = train_clf(X_xrp, y_xrp)

# ── Live Feature for Today ──
def today_feat(df):
    i = len(df)-1
    p   = df["price"].iat[i]
    ret = df["return"].iat[i]
    vol = df["vol14"].iat[i]
    return [[
        df["rsi"].iat[i],
        vol,
        p - df["ema50"].iat[i],
        df["sma5"].iat[i] - df["sma20"].iat[i],
        ret
    ]]

p_btc = clf_btc.predict_proba(today_feat(btc_hist))[0][1]
p_xrp = clf_xrp.predict_proba(today_feat(xrp_hist))[0][1]

use_ml_btc = p_btc >= CLASS_PROB_THRESH
use_ml_xrp = p_xrp >= CLASS_PROB_THRESH

# ── Sidebar: Allocation ↔ Conversion & Min Order
st.sidebar.title("💰 Investment Settings")
usd_total = st.sidebar.number_input("Total Investment ($)",100.0,1e6,3000.0,100.0)
pct_btc   = st.sidebar.slider("BTC Allocation (%)",0,100,70)
usd_btc   = usd_total * pct_btc/100
usd_xrp   = usd_total - usd_btc
gbp_rate  = st.sidebar.number_input("GBP/USD Rate",1.1,1.6,1.27,0.01)
st.sidebar.metric("Alloc USD/GBP", f"${usd_total:,.2f}", f"£{usd_total/gbp_rate:,.2f}")
min_order = st.sidebar.number_input("Min Order (BTC)",1e-6,1e-2,5e-4,1e-6,format="%.6f")
MIN_ORDER = max(min_order, (usd_btc/GRID_MAX)/btc_p if btc_p else 0)
st.sidebar.caption(f"🔒 Min Order ≥ {MIN_ORDER:.6f} BTC (~${MIN_ORDER*btc_p:.2f})")

# ── Display Final Defaults ──
st.sidebar.markdown("### ⚙️ Final Defaults")
st.sidebar.write(f"BTC{' (ML)' if use_ml_btc else ''}: RSI<{btc_params[0]}, TP×{btc_params[1]}, SL{btc_params[2]}% (p={p_btc:.0%})")
st.sidebar.write(f"XRP{' (ML)' if use_ml_xrp else ''}: Mean{int(xrp_params[0])}d, Bounce{xrp_params[1]}%, SL{xrp_params[2]}%, Dip{xrp_params[3]}% (p={p_xrp:.0%})")

# ── Grid Calculation & Display
def display_bot(name,p,drop,levels,ml_flag,alloc,is_btc):
    st.header(name)
    unit = "USD" if is_btc else "BTC"
    st.write(f"- Price: {p:.6f} {unit}")
    if drop>0:
        if ml_flag:
            st.success("✅ ML Override Active")
        st.write(f"- Drop %: {drop:.2f}%")
        bot = p * (1-drop/100)
        step = (p-bot)/levels
        per  = (alloc/p)/levels if is_btc else (alloc/(p*btc_p))/levels
        st.write(f"- Lower: {bot:.6f}; Upper: {p:.6f}; Step: {step:.6f}")
        st.write(f"- Per-Order: {per:.6f} BTC {'✅' if per>=MIN_ORDER else '❌'}")
        tp = p*(1+drop/100) if is_btc else p*(1+xrp_params[1]/100)
        st.write(f"- Take-Profit: {tp:.6f} {unit}")
        st.write("🔄 Redeploy Bot now")
    else:
        st.error("🛑 Terminate Bot")

vol14 = btc_hist["vol14"].iat[-1]
ret24= btc_ch if btc_ch is not None else btc_hist["return"].iat[-1]
drop_btc = vol14 if ret24<vol14 else (2*vol14 if ret24>2*vol14 else ret24)
levels_btc = GRID_PRIMARY if not use_ml_btc else GRID_MAX
display_bot("🟡 BTC/USDT Bot", btc_p, drop_btc, levels_btc, use_ml_btc, usd_btc, True)

sig_xrp = (
    xrp_hist["price"].iat[-1] < xrp_hist["price"].rolling(int(xrp_params[0])).mean().iat[-1]
) and (
    xrp_hist["vol14"].iat[-1] > xrp_hist["vol14"].iat[-2]
)
drop_xrp = xrp_params[1] if sig_xrp else 0
levels_xrp = GRID_PRIMARY if not use_ml_xrp else GRID_MAX
display_bot("🟣 XRP/BTC Bot", xrp_p, drop_xrp, levels_xrp, use_ml_xrp, usd_xrp, False)

with st.expander("ℹ️ About"):
    st.markdown("""
    • **Classification RF** tuned via GridSearchCV when enough samples exist.  
    • **Override defaults** only when p(win) ≥ 80 %.  
    • All original indicators, backtests, grid logic, deploy/terminate signals remain.
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
