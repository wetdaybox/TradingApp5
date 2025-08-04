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
import concurrent.futures

# ── Auto‐refresh every 60 s ──
st_autorefresh(interval=60_000, key="refresh")

# ── Page Setup ──
st.set_page_config(layout="centered")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
now = datetime.now(pytz.timezone("Europe/London"))
st.caption(f"Last updated: {now:%Y-%m-%d %H:%M %Z}")

# ── Deployment Flags ──
if "deployed_b" not in st.session_state:
    st.session_state.deployed_b = False
if "deployed_x" not in st.session_state:
    st.session_state.deployed_x = False

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
    return {}

@st.cache_data(ttl=600)
def load_history(coin, vs):
    js     = fetch_json(
                f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
                {"vs_currency": vs, "days": HISTORY_DAYS}
             ) or {}
    prices = js.get("prices", [])
    df     = pd.DataFrame(prices, columns=["ts","price"])
    if df.empty:
        return df
    df["date"]   = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change() * 100
    df["ema50"]  = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    df["sma5"]   = df["price"].rolling(5).mean()
    df["sma20"]  = df["price"].rolling(20).mean()
    df["vol14"]  = df["return"].rolling(VOL_WINDOW).std().fillna(np.nan)
    delta        = df["price"].diff()
    gain         = delta.clip(lower=0)
    loss         = -delta.clip(upper=0)
    df["rsi"]    = 100 - 100 / (
                       1 + gain.rolling(RSI_WINDOW).mean()/
                           loss.rolling(RSI_WINDOW).mean()
                   )
    return df

@st.cache_data(ttl=60)
def load_live():
    def one(id, vs, extra):
        return fetch_json(
            "https://api.coingecko.com/api/v3/simple/price",
            {"ids":id, "vs_currencies":vs, **extra}
        ) or {}
    with concurrent.futures.ThreadPoolExecutor() as ex:
        f1 = ex.submit(one, "bitcoin", "usd", {"include_24hr_change":"true"})
        f2 = ex.submit(one, "ripple",  "btc", {})
        j1, j2 = f1.result(), f2.result()
    btc = j1.get("bitcoin", {})
    xrp = j2.get("ripple", {})
    return {
        "BTC": (btc.get("usd", np.nan), btc.get("usd_24h_change", np.nan)),
        "XRP": (xrp.get("btc", np.nan), None)
    }

# ── Signal Generators ──
def gen_signals(df, is_btc, params):
    X,y = [],[]
    for i in range(EMA_TREND, len(df)-1):
        p = df["price"].iat[i]
        ret = df["return"].iat[i]
        vol = df["vol14"].iat[i]
        ema_diff = p - df["ema50"].iat[i]
        mom = df["sma5"].iat[i] - df["sma20"].iat[i]
        rsi = df["rsi"].iat[i]
        if is_btc:
            rsi_th, tp_m, sl = params
            cond = (ema_diff>0) and (mom>0) and (rsi<rsi_th) and (ret>=vol)
        else:
            m,b,sl,dip = params
            mval = df["price"].rolling(m).mean().iat[i]
            cond = (p<mval) and (((mval-p)/p*100)>=dip) and (vol>df["vol14"].iat[i-1])
        if not cond:
            continue
        X.append([rsi,vol,ema_diff,mom,ret])
        y.append(1 if df["price"].iat[i+1]>p else 0)
    return np.array(X), np.array(y)

@st.cache_resource
def train_models(Xb,yb,Xx,yx):
    def t(X,y):
        if len(y)>=6 and len(np.unique(y))>1:
            gs = GridSearchCV(RandomForestClassifier(random_state=0),
                              {"n_estimators":[50,100],"max_depth":[3,5]},
                              cv=3, scoring="accuracy", n_jobs=1)
            gs.fit(X,y)
            return gs.best_estimator_
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        if len(y)>0:
            clf.fit(X,y)
        return clf
    return t(Xb,yb), t(Xx,yx)

def today_feat(df):
    if df.empty: return None
    i = len(df)-1
    return [[
        df["rsi"].iat[i],
        df["vol14"].iat[i],
        df["price"].iat[i]-df["ema50"].iat[i],
        df["sma5"].iat[i]-df["sma20"].iat[i],
        df["return"].iat[i],
    ]]

def safe_prob(clf, feat):
    if feat is None: return 0.0
    probs = clf.predict_proba(feat)[0]
    return probs[1] if len(probs)>1 else 0.0

# ── Initialization ──
with st.spinner("🚀 Loading data…"):
    btc_hist = load_history("bitcoin","usd")
    xrp_hist = load_history("ripple","btc")
    live     = load_live()
    btc_p,btc_ch = live["BTC"]
    xrp_p,_     = live["XRP"]

    btc_params = (75,1.5,1.0)
    xrp_params = (10,75,50,1.0)

    Xb,yb = gen_signals(btc_hist, True, btc_params)
    Xx,yx = gen_signals(xrp_hist, False, xrp_params)
    clf_b,clf_x = train_models(Xb,yb,Xx,yx)

    p_b = safe_prob(clf_b, today_feat(btc_hist))
    p_x = safe_prob(clf_x, today_feat(xrp_hist))

# ── Sidebar ──
st.sidebar.header("💰 Investment Settings")
usd_tot = st.sidebar.number_input("Total Investment ($)",100.0,1e6,3000.0,100.0)
pct_btc = st.sidebar.slider("BTC Allocation (%)",0,100,70)
usd_btc = usd_tot * pct_btc/100
usd_xrp = usd_tot - usd_btc
gbp_rate= st.sidebar.number_input("GBP/USD Rate",1.10,1.60,1.27,0.01)
st.sidebar.metric("Total Value (USD/GBP)",f"${usd_tot:,.2f}",f"£{usd_tot/gbp_rate:,.2f}")
min_ord = st.sidebar.number_input("Min Order (BTC)",1e-6,1e-2,5e-4,1e-6,format="%.6f")
MIN_ORDER= max(min_ord,(usd_btc/GRID_MAX)/btc_p if btc_p else 0)
st.sidebar.caption(f"Min Order ≥ {MIN_ORDER:.6f} BTC (~${MIN_ORDER*btc_p:.2f})")

# ── Bespoke Grid Sliders ──
default_b = GRID_MAX if p_b>=CLASS_PROB_THRESH else GRID_PRIMARY
default_x = GRID_MAX if p_x>=CLASS_PROB_THRESH else GRID_PRIMARY
g_b = st.sidebar.slider("BTC Grid Levels",5,GRID_MAX,default_b,key="gb")
g_x = st.sidebar.slider("XRP Grid Levels",5,GRID_MAX,default_x,key="gx")

# ── Compute Drop & Actions ──
def compute_drop(df, price, change):
    if df.empty: return 0
    vol = df["vol14"].iat[-1]
    ret = change if change is not None else df["return"].iat[-1]
    if np.isnan(vol) or ret<vol: return 0
    return vol if ret<=2*vol else 2*vol

drop_b = compute_drop(btc_hist,btc_p,btc_ch)
low_b  = btc_p*(1-drop_b/100); up_b=btc_p; tp_b=up_b*(1+drop_b/100)
if not st.session_state.deployed_b:
    act_b = "Not Deployed"
elif drop_b>0:
    act_b = "Redeploy"
elif btc_p>=tp_b:
    act_b = "Take-Profit"
else:
    act_b = "Hold"

drop_x = compute_drop(xrp_hist,xrp_p,None)
low_x  = xrp_p*(1-drop_x/100); up_x=xrp_p; tp_x=up_x*(1+drop_x/100)
if not st.session_state.deployed_x:
    act_x = "Not Deployed"
elif drop_x>0:
    act_x = "Redeploy"
elif xrp_p>=tp_x:
    act_x = "Take-Profit"
else:
    act_x = "Hold"

# ── Display Function ──
def show_bot(title, grids, low, up, tp, act, key):
    st.subheader(title)

    if act=="Not Deployed":
        st.warning("⚠️ Bot not yet deployed.  Click 🔄 Redeploy Now to start grid trading.")
        if st.button("🔄 Redeploy Now", key=f"{key}_deploy_first"):
            if key=="b": st.session_state.deployed_b = True
            else:        st.session_state.deployed_x = True
            st.success("✅ Bot Deployed")
        return

    if act=="Take-Profit":
        st.success(f"💰 TAKE-PROFIT: price ≥ {up:,.6f}")
        return

    if act=="Hold":
        st.info("⏸ HOLD: no action needed right now.")
        return

    # Redeploy case
    c1,c2 = st.columns(2)
    with c1:
        st.metric("Grids",f"{grids}")
        st.metric("Lower",f"{low:,.6f}")
        st.metric("Upper",f"{up:,.6f}")
        st.metric("TP",   f"{tp:,.6f}")
    with c2:
        if st.button("🔄 Redeploy Now",key=f"{key}_r"):
            if key=="b": st.session_state.deployed_b = True
            else:        st.session_state.deployed_x = True
            st.success("✅ Copy into Crypto.com Grid Box")

    with st.expander("Details"):
        hist = btc_hist if "BTC" in title else xrp_hist
        prob = p_b if "BTC" in title else p_x
        vl   = hist["vol14"].iat[-1] if len(hist)>=VOL_WINDOW else None
        rs   = hist["rsi"].iat[-1]   if len(hist)>=RSI_WINDOW else None
        st.write(f"- **Action:** {act}")
        if vl is not None: st.write(f"- 14d Vol: {vl:.2f}%")
        if rs is not None: st.write(f"- 14d RSI: {rs:.1f}")
        st.write(f"- ML Conf: {int(prob*100)}%")
        st.write(f"- Grids used: {grids}")

show_bot("🟡 BTC/USDT Bot", g_b, low_b, up_b, tp_b, act_b, "b")
show_bot("🟣 XRP/BTC Bot", g_x, low_x, up_x, tp_x, act_x, "x")

# ── About & Requirements ──
with st.expander("ℹ️ How to use"):
    st.write("""
    1. If you see ⚠️ Not Deployed, click 🔄 Redeploy Now to start.  
    2. If you see 🔄 Redeploy, copy Grids/Lower/Upper into Crypto.com Grid Bot.  
    3. If you see 💰 Take-Profit, terminate your bot and close all to realize gains.  
    4. If you see ⏸ Hold, leave running until next signal.  
    5. Use the sidebar to adjust allocation, GBP rate, min‐order, and grid counts.
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
