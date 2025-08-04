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

# ── Auto-refresh every 60 s ──
st_autorefresh(interval=60_000, key="refresh")

# ── Page setup ──
st.set_page_config(layout="centered")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
now = datetime.now(pytz.timezone("Europe/London"))
st.caption(f"Last updated: {now:%Y-%m-%d %H:%M %Z}")

# ── Persisted state for deploy/terminate ──
for bot in ("b", "x"):
    if f"deployed_{bot}" not in st.session_state:
        st.session_state[f"deployed_{bot}"] = False
    if f"terminated_{bot}" not in st.session_state:
        st.session_state[f"terminated_{bot}"] = False

# ── Constants ──
HISTORY_DAYS      = 90
VOL_WINDOW        = 14
RSI_WINDOW        = 14
EMA_TREND         = 50
RSI_OVERBOUGHT    = 75
GRID_PRIMARY      = 20
GRID_MAX          = 30
MIN_VOL_THRESHOLD = 1.0  # percent
CLASS_PROB_THRESH = 0.80
MAX_RETRIES       = 3

# ── Fetch/Caching ──
def fetch_json(url, params):
    for i in range(MAX_RETRIES):
        r = requests.get(url, params=params, timeout=10)
        if r.status_code==429:
            time.sleep(2**i)
            continue
        r.raise_for_status()
        return r.json()
    return {}

@st.cache_data(ttl=600)
def load_history(coin, vs):
    js = fetch_json(
        f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
        {"vs_currency":vs, "days":HISTORY_DAYS}
    ) or {}
    prices = js.get("prices",[])
    df = pd.DataFrame(prices, columns=["ts","price"])
    if df.empty: return df
    df["date"]   = pd.to_datetime(df["ts"],unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change()*100
    df["ema50"]  = df["price"].ewm(span=EMA_TREND,adjust=False).mean()
    df["sma5"]   = df["price"].rolling(5).mean()
    df["sma20"]  = df["price"].rolling(20).mean()
    df["vol14"]  = df["return"].rolling(VOL_WINDOW).std().fillna(np.nan)
    d = df["price"].diff()
    df["rsi"]    = 100-100/(1 + 
                      d.clip(lower=0).rolling(RSI_WINDOW).mean()/ 
                      (-d.clip(upper=0)).rolling(RSI_WINDOW).mean())
    return df

@st.cache_data(ttl=60)
def load_live():
    def one(id,vs,extra):
        return fetch_json(
            "https://api.coingecko.com/api/v3/simple/price",
            {"ids":id,"vs_currencies":vs,**extra}
        ) or {}
    with concurrent.futures.ThreadPoolExecutor() as ex:
        f1 = ex.submit(one,"bitcoin","usd",{"include_24hr_change":"true"})
        f2 = ex.submit(one,"ripple","btc",{})
        j1,j2 = f1.result(), f2.result()
    b = j1.get("bitcoin",{})
    x = j2.get("ripple",{})
    return {
        "BTC":(b.get("usd",np.nan),b.get("usd_24h_change",np.nan)),
        "XRP":(x.get("btc",np.nan),None)
    }

# ── Data & ML prep (unchanged) ──
def gen_signals(df,is_btc,params):
    X,y = [],[]
    for i in range(EMA_TREND,len(df)-1):
        p,ret,vol = df["price"].iat[i],df["return"].iat[i],df["vol14"].iat[i]
        ed = p-df["ema50"].iat[i]
        mo = df["sma5"].iat[i]-df["sma20"].iat[i]
        rs = df["rsi"].iat[i]
        if is_btc:
            rsi_th,_,_ = params
            cond = ed>0 and mo>0 and rs<rsi_th and ret>=vol
        else:
            m,b,_,dip = params
            mv = df["price"].rolling(m).mean().iat[i]
            cond = p<mv and ((mv-p)/p*100)>=dip and vol>df["vol14"].iat[i-1]
        if not cond: continue
        X.append([rs,vol,ed,mo,ret])
        y.append(1 if df["price"].iat[i+1]>p else 0)
    return np.array(X),np.array(y)

@st.cache_resource
def train_models(Xb,yb,Xx,yx):
    def t(X,y):
        if len(y)>=6 and len(np.unique(y))>1:
            gs = GridSearchCV(RandomForestClassifier(random_state=0),
                              {"n_estimators":[50,100],"max_depth":[3,5]},
                              cv=3, scoring="accuracy", n_jobs=1)
            gs.fit(X,y)
            return gs.best_estimator_
        clf = RandomForestClassifier(n_estimators=100,random_state=0)
        if len(y)>0: clf.fit(X,y)
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

def safe_prob(clf,feat):
    if feat is None: return 0.0
    p = clf.predict_proba(feat)[0]
    return p[1] if len(p)>1 else 0.0

# ── Load & train ──
with st.spinner("Loading data…"):
    btc_hist = load_history("bitcoin","usd")
    xrp_hist = load_history("ripple","btc")
    live     = load_live()
    btc_p,btc_ch = live["BTC"]
    xrp_p,_     = live["XRP"]

    btc_params = (75,1.5,1.0)
    xrp_params = (10,75,50,1.0)
    Xb,yb = gen_signals(btc_hist, True,  btc_params)
    Xx,yx = gen_signals(xrp_hist, False, xrp_params)
    clf_b,clf_x = train_models(Xb,yb,Xx,yx)
    p_b = safe_prob(clf_b, today_feat(btc_hist))
    p_x = safe_prob(clf_x, today_feat(xrp_hist))

# ── Sidebar ──
st.sidebar.header("💰 Investment Settings")
usd_tot  = st.sidebar.number_input("Total Investment ($)",100.0,1e6,3000.0,100.0)
pct_btc  = st.sidebar.slider("BTC Allocation (%)",0,100,70)
usd_btc  = usd_tot*pct_btc/100
usd_xrp  = usd_tot-usd_btc
gbp_rate = st.sidebar.number_input("GBP/USD Rate",1.10,1.60,1.27,0.01)
st.sidebar.metric("Portfolio",f"${usd_tot:,.2f}",f"£{usd_tot/gbp_rate:,.2f}")
min_ord  = st.sidebar.number_input("Min Order (BTC)",1e-6,1e-2,5e-4,1e-6,format="%.6f")
MIN_ORDER= max(min_ord,(usd_btc/GRID_MAX)/btc_p if btc_p else 0)
st.sidebar.caption(f"Min Order ≥ {MIN_ORDER:.6f} BTC (~${MIN_ORDER*btc_p:.2f})")

# ── Grid counts ──
default_b = GRID_MAX if p_b>=CLASS_PROB_THRESH else GRID_PRIMARY
default_x = GRID_MAX if p_x>=CLASS_PROB_THRESH else GRID_PRIMARY
g_b = st.sidebar.slider("BTC Grid Levels",5,GRID_MAX,default_b,key="gb")
g_x = st.sidebar.slider("XRP Grid Levels",5,GRID_MAX,default_x,key="gx")

# ── Compute regime filters & drop logic ──
def regime_ok(df):
    if df.empty: return False
    vl  = df["vol14"].iat[-1]
    p   = df["price"].iat[-1]
    eok = p > df["ema50"].iat[-1]
    mok = df["sma5"].iat[-1] > df["sma20"].iat[-1]
    rs  = df["rsi"].iat[-1]
    rok = rs < RSI_OVERBOUGHT
    vok = vl >= MIN_VOL_THRESHOLD
    return eok and mok and rok and vok

def compute_drop(df, price, change):
    if df.empty: return 0
    vol = df["vol14"].iat[-1]
    ret = change if change is not None else df["return"].iat[-1]
    if np.isnan(vol) or ret<vol: return 0
    return vol if ret<=2*vol else 2*vol

# BTC action
drop_b = compute_drop(btc_hist, btc_p, btc_ch)
low_b, up_b = btc_p*(1-drop_b/100), btc_p
tp_b = up_b*(1+drop_b/100)
if st.session_state.terminated_b:
    if regime_ok(btc_hist):
        st.session_state.terminated_b = False
        act_b = "Not Deployed"
    else:
        act_b = "Terminated"
elif not regime_ok(btc_hist):
    act_b = "Terminate"
elif not st.session_state.deployed_b:
    act_b = "Not Deployed"
elif drop_b>0:
    act_b = "Redeploy"
elif btc_p>=tp_b:
    act_b = "Take-Profit"
else:
    act_b = "Hold"

# XRP action
drop_x = compute_drop(xrp_hist, xrp_p, None)
low_x, up_x = xrp_p*(1-drop_x/100), xrp_p
tp_x = up_x*(1+drop_x/100)
if st.session_state.terminated_x:
    if regime_ok(xrp_hist):
        st.session_state.terminated_x = False
        act_x = "Not Deployed"
    else:
        act_x = "Terminated"
elif not regime_ok(xrp_hist):
    act_x = "Terminate"
elif not st.session_state.deployed_x:
    act_x = "Not Deployed"
elif drop_x>0:
    act_x = "Redeploy"
elif xrp_p>=tp_x:
    act_x = "Take-Profit"
else:
    act_x = "Hold"

# ── UI Renderer ──
def show_bot(title, grids, low, up, tp, act, key):
    st.subheader(title)
    st.metric("Grids",       f"{grids}")
    st.metric("Lower Price", f"{low:,.6f}")
    st.metric("Upper Price", f"{up:,.6f}")
    st.metric("Take-Profit", f"{tp:,.6f}")

    if act=="Terminated":
        st.error("🛑 Bot Terminated—waiting for ideal regime.")
        return
    if act=="Terminate":
        if st.button("🛑 Terminate Bot",key=f"{key}_term"):
            st.session_state[f"terminated_{key}"] = True
        return
    if act=="Not Deployed":
        st.warning("⚠️ Bot not yet deployed. Click 🔄 Deploy to start.")
        if st.button("🔄 Deploy", key=f"{key}_deploy"):
            st.session_state[f"deployed_{key}"] = True
            st.success("✅ Bot Deployed")
        return
    if act=="Take-Profit":
        st.success(f"💰 TAKE-PROFIT: price ≥ {up:,.6f}")
        return
    if act=="Hold":
        st.info("⏸ HOLD: no action needed right now.")
        return
    # Redeploy
    st.info("🔔 Grid Reset Signal")
    if st.button("🔄 Redeploy Now",key=f"{key}_redeploy"):
        # stay deployed
        st.success("✅ Copy these parameters into Crypto.com Grid Box")

# ── Render both bots ──
show_bot("🟡 BTC/USDT Bot", g_b, low_b, up_b, tp_b, act_b, "b")
show_bot("🟣 XRP/BTC Bot", g_x, low_x, up_x, tp_x, act_x, "x")

# ── Instructions & requirements ──
with st.expander("ℹ️ How to use"):
    st.write("""
    1. **Terminate** if regime breaks (🛑). Bot will wait until EMA/momentum/RSI/vol are all OK.  
    2. **Deploy** your first grid after regime is OK (🔄 Deploy).  
    3. **Redeploy** on dip-signals (🔔). Copy Grids/Lower/Upper into Crypto.com.  
    4. **Take-Profit** when price ≥ Upper (💰). Terminate/Close All on exchange.  
    5. **Hold** otherwise (⏸). Let your grid run.
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
