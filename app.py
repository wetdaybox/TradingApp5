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
st_autorefresh(interval=60_000, key="refresh")

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

# ── Fetch Helpers ──
def fetch_json(url, params):
    for i in range(MAX_RETRIES):
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 429:
            time.sleep(2**i)
            continue
        r.raise_for_status()
        return r.json()
    return {}

@st.cache_data(ttl=600)
def load_history(coin, vs):
    js = fetch_json(
        f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
        {"vs_currency": vs, "days": HISTORY_DAYS}
    ) or {}
    prices = js.get("prices", [])
    df = pd.DataFrame(prices, columns=["ts","price"])
    if df.empty: 
        return df
    df["date"]   = pd.to_datetime(df["ts"],unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change()*100
    df["ema50"]  = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    df["sma5"]   = df["price"].rolling(5).mean()
    df["sma20"]  = df["price"].rolling(20).mean()
    df["vol14"]  = df["return"].rolling(VOL_WINDOW).std().fillna(0)
    delta       = df["price"].diff()
    gain        = delta.clip(lower=0)
    loss        = -delta.clip(upper=0)
    df["rsi"]   = 100 - 100 / (1 + gain.rolling(RSI_WINDOW).mean()/loss.rolling(RSI_WINDOW).mean())
    return df.dropna()

@st.cache_data(ttl=60)
def load_live():
    js  = fetch_json(
        "https://api.coingecko.com/api/v3/simple/price",
        {"ids":"bitcoin,ripple","vs_currencies":"usd,btc","include_24hr_change":"true"}
    ) or {}
    btc = js.get("bitcoin",{})
    xrp = js.get("ripple",{})
    return {
        "BTC": (btc.get("usd",np.nan), btc.get("usd_24h_change",np.nan)),
        "XRP": (xrp.get("btc",np.nan), None)
    }

# ── Signal & ML logic ──
def gen_signals(df, is_btc, params):
    X,y=[],[]
    for i in range(EMA_TREND,len(df)-1):
        p,ret,vol = df["price"].iat[i],df["return"].iat[i],df["vol14"].iat[i]
        ema_diff   = p-df["ema50"].iat[i]
        mom        = df["sma5"].iat[i]-df["sma20"].iat[i]
        rsi        = df["rsi"].iat[i]
        if is_btc:
            rsi_th,_,_ = params
            cond = (ema_diff>0) and (mom>0) and (rsi<rsi_th) and (ret>=vol)
        else:
            m,b,_,dip = params
            mval = df["price"].rolling(m).mean().iat[i]
            cond = (p<mval) and (((mval-p)/p*100)>=dip) and (vol>df["vol14"].iat[i-1])
        if not cond: continue
        X.append([rsi,vol,ema_diff,mom,ret])
        profit = df["price"].iat[i+1]-p
        y.append(1 if profit>0 else 0)
    return np.array(X),np.array(y)

def train_clf(X,y):
    if len(y)>=6 and len(np.unique(y))>1:
        gs = GridSearchCV(RandomForestClassifier(random_state=0),
                          {"n_estimators":[50,100],"max_depth":[3,5]},
                          cv=3, scoring="accuracy", n_jobs=-1)
        gs.fit(X,y)
        return gs.best_estimator_
    clf = RandomForestClassifier(n_estimators=100,random_state=0)
    if len(y)>0: clf.fit(X,y)
    return clf

def today_feat(df):
    if df.empty: return None
    i = len(df)-1
    return [[ df["rsi"].iat[i],
              df["vol14"].iat[i],
              df["price"].iat[i]-df["ema50"].iat[i],
              df["sma5"].iat[i]-df["sma20"].iat[i],
              df["return"].iat[i] ]]

def safe_prob(clf,feat):
    if feat is None: return 0.0
    probs=clf.predict_proba(feat)[0]
    return probs[1] if probs.shape[0]>1 else 0.0

# ── Load & train ──
btc_hist = load_history("bitcoin","usd")
xrp_hist = load_history("ripple","btc")
(live_btc_price, live_btc_ch), (live_xrp_price,_) = load_live().values()

btc_params = (75,1.5,1.0)
xrp_params = (10,75,50,1.0)

Xb,yb = gen_signals(btc_hist,True, btc_params)
Xx,yx = gen_signals(xrp_hist,False,xrp_params)

clf_btc = train_clf(Xb,yb)
clf_xrp = train_clf(Xx,yx)

p_btc   = safe_prob(clf_btc, today_feat(btc_hist))
p_xrp   = safe_prob(clf_xrp, today_feat(xrp_hist))
use_btc = p_btc>=CLASS_PROB_THRESH
use_xrp = p_xrp>=CLASS_PROB_THRESH

# ── Sidebar ──
st.sidebar.header("💰 Investment")
usd_total     = st.sidebar.number_input("Total $",100.0,1e6,3000.0,100.0)
pct_btc       = st.sidebar.slider("BTC %",0,100,70)
usd_btc_alloc = usd_total*pct_btc/100
usd_xrp_alloc = usd_total-usd_btc_alloc
gbp_rate      = st.sidebar.number_input("GBP/USD",1.1,1.6,1.27,0.01)
st.sidebar.metric("Value (USD/GBP)",f"${usd_total:,.2f}",f"£{usd_total/gbp_rate:,.2f}")
min_order     = st.sidebar.number_input("Min Order (BTC)",1e-6,1e-2,5e-4,1e-6,format="%.6f")
MIN_ORDER     = max(min_order, (usd_btc_alloc/GRID_MAX)/live_btc_price if live_btc_price else 0)
st.sidebar.caption(f"Min Order ≥ {MIN_ORDER:.6f} BTC (~${MIN_ORDER*live_btc_price:.2f})")

# ── Action logic ──
def compute_drop(df, live_price, live_change):
    if df.empty: return 0
    vol14 = df["vol14"].iat[-1]
    ret   = live_change if live_change is not None else df["return"].iat[-1]
    if ret < vol14: return 0
    return vol14 if ret<=2*vol14 else 2*vol14

drop_btc = compute_drop(btc_hist, live_btc_price, live_btc_ch)
drop_xrp = xrp_params[1] if not xrp_hist.empty and \
    (xrp_hist["price"].iat[-1] < xrp_hist["price"].rolling(int(xrp_params[0])).mean().iat[-1] \
     and xrp_hist["vol14"].iat[-1]>xrp_hist["vol14"].iat[-2]) else 0

# ── Display function ──
def show_bot(name, price, drop, is_btc, use_ml, usd_alloc):
    unit = "USD" if is_btc else "BTC"
    st.subheader(f"{name}")
    st.write(f"**Price:** {price:,.6f} {unit}")
    action = "⏸ Hold"
    if drop>0:
        action = "🔄 Redeploy Now"
    elif drop==0:
        action = "🛑 Terminate Bot"
    st.markdown(f"**Action:** {action}")
    if drop>0:
        tp = price*(1+drop/100)
        st.write(f"**Take-Profit:** {tp:,.6f} {unit}")

    with st.expander("Details"):
        st.write(f"- ML Override: {'Yes' if use_ml else 'No'} (p={int((p_btc if is_btc else p_xrp)*100)}%)")
        st.write(f"- Grid Depth (Drop %): {drop:.2f}%")
        low  = price*(1-drop/100)
        step = (price-low)/(GRID_PRIMARY if not use_ml else GRID_MAX)
        st.write(f"- Bounds: {low:.6f} ↔ {price:.6f}")
        st.write(f"- Step: {step:.6f}")
        per = (usd_alloc/price)/(GRID_PRIMARY if not use_ml else GRID_MAX)
        st.write(f"- Per-Order: {per:.6f} BTC  {'✅' if per>=MIN_ORDER else '❌'}")

show_bot("🟡 BTC/USDT Bot", live_btc_price, drop_btc, True,  use_btc, usd_btc_alloc)
show_bot("🟣 XRP/BTC Bot", live_xrp_price, drop_xrp, False, use_xrp, usd_xrp_alloc)

# ── Hidden About & Requirements ──
with st.expander("ℹ️ About & How-To"):
    st.write("""
    **Quickstart**  
    1. Set **Total $** and **BTC %** in the sidebar.  
    2. Sidebar shows **Min Order** you must meet.  
    3. On each Bot:  
       - **Action:** one of Redeploy Now / Terminate / Hold  
       - **Take-Profit** appears when Redeploy is active.  
    4. Click **Redeploy Now** in your Crypto.com Grid setup, or **Terminate Bot**  
       when instructed.  
    5. Use the **Details** expander for full parameter values.  
    6. App refreshes every 60 s with new live data.
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
