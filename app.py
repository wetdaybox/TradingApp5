# app.py
import os, shutil, pickle, time, random
from datetime import datetime
import requests, concurrent.futures
import pandas as pd, numpy as np
import streamlit as st
from sklearn.linear_model import SGDClassifier
from streamlit_autorefresh import st_autorefresh
import pytz

# ───────── Persistence Helpers ─────────
STATE_DIR   = ".state"
ML_BUF_FILE = os.path.join(STATE_DIR, "ml_buffer.pkl")
FLAGS_FILE  = os.path.join(STATE_DIR, "flags.pkl")

def ensure_state_dir():
    if not os.path.isdir(STATE_DIR):
        os.makedirs(STATE_DIR)

def load_flags():
    ensure_state_dir()
    if os.path.isfile(FLAGS_FILE):
        return pickle.load(open(FLAGS_FILE,"rb"))
    return {"deployed_b":False,"terminated_b":False,
            "deployed_x":False,"terminated_x":False,
            "bal_b":None,"bal_x":None}

def save_flags(f):
    pickle.dump(f, open(FLAGS_FILE,"wb"))

def load_ml_buffer():
    ensure_state_dir()
    if os.path.isfile(ML_BUF_FILE):
        return pickle.load(open(ML_BUF_FILE,"rb"))
    return {"X":[], "y":[], "ts":[]}

def save_ml_buffer(b):
    pickle.dump(b, open(ML_BUF_FILE,"wb"))

def persist_all():
    save_flags({
        "deployed_b":st.session_state.deployed_b,
        "terminated_b":st.session_state.terminated_b,
        "deployed_x":st.session_state.deployed_x,
        "terminated_x":st.session_state.terminated_x,
        "bal_b":st.session_state.bal_b,
        "bal_x":st.session_state.bal_x
    })
    save_ml_buffer({
        "X":st.session_state.mem_X,
        "y":st.session_state.mem_y,
        "ts":st.session_state.mem_ts
    })

flags  = load_flags()
ml_buf = load_ml_buffer()

# ───────── Streamlit Setup ─────────
st.set_page_config(layout="centered")
st_autorefresh(interval=60_000, key="refresh")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
st.caption(f"Last updated: {datetime.now(pytz.timezone('Europe/London')):%Y-%m-%d %H:%M %Z}")

for k,v in flags.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("mem_X", ml_buf["X"])
st.session_state.setdefault("mem_y", ml_buf["y"])
st.session_state.setdefault("mem_ts",ml_buf["ts"])

if "online_clf" not in st.session_state:
    clf = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
    clf.partial_fit(np.zeros((2,5)), [0,1], classes=[0,1])
    st.session_state.online_clf = clf

# ───────── Robust fetch_json ─────────
MAX_RETRIES = 3
def fetch_json(url, params):
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException:
            time.sleep(2**i)
    return None

# ───────── Sidebar Inputs ─────────
st.sidebar.header("💰 Strategy Settings")
usd_tot   = st.sidebar.number_input("Total Investment ($)",100.0,1e6,3000.0,100.0)
pct_btc   = st.sidebar.slider("BTC Allocation (%)",0,100,70)
usd_btc   = usd_tot * pct_btc/100
usd_xrp   = usd_tot - usd_btc
gbp_rate  = st.sidebar.number_input("GBP/USD Rate",1.10,1.60,1.27,0.01)
st.sidebar.metric("Portfolio", f"${usd_tot:,.2f}", f"£{usd_tot/gbp_rate:,.2f}")
stop_loss = st.sidebar.slider("Stop-Loss (%)",1.0,5.0,2.0,0.1)
compound  = st.sidebar.checkbox("Enable Compounding", value=False)
mode      = st.sidebar.radio("Mode",["Start New Cycle","Continue Existing"],
                             index=0 if st.session_state.get("mode","new")=="new" else 1)
st.session_state.mode = "new" if mode=="Start New Cycle" else "cont"

# ─── Manual-Override Fix ───
override = st.sidebar.checkbox("Manual Grid Override", value=False)
manual_b = st.sidebar.number_input("BTC/USDT Grids",2,30,6) if (override and st.session_state.mode=="new") else None
manual_x = st.sidebar.number_input("XRP/BTC Grids",2,30,8) if (override and st.session_state.mode=="new") else None

# ── Persistence & Resets ──
st.sidebar.header("⚙️ Persistence & Resets")
if st.sidebar.button("🔄 Reset Bot State"):
    for b in ("b","x"):
        st.session_state[f"deployed_{b}"]=False
        st.session_state[f"terminated_{b}"]=False
    st.sidebar.success("Bot state reset.")
if st.sidebar.button("🔄 Reset Balances"):
    st.session_state.bal_b=None; st.session_state.bal_x=None
    st.sidebar.success("Balances reset.")
if st.sidebar.button("🔄 Clear ML Memory"):
    st.session_state.mem_X=[]; st.session_state.mem_y=[]; st.session_state.mem_ts=[]
    st.sidebar.success("ML memory cleared.")
if st.sidebar.button("🗑️ Delete Everything"):
    shutil.rmtree(STATE_DIR, ignore_errors=True)
    for k in ("deployed_b","terminated_b","deployed_x","terminated_x","bal_b","bal_x","mem_X","mem_y","mem_ts"):
        st.session_state.pop(k,None)
    st.sidebar.success("All state deleted.")
    st.experimental_rerun()

if st.session_state.bal_b is None: st.session_state.bal_b=usd_btc
if st.session_state.bal_x is None: st.session_state.bal_x=usd_xrp

# ── Data Loading & Indicators ──
H_DAYS, VOL_W, RSI_W, EMA_T = 90,14,14,50
BASE_RSI_OB, MIN_VOL       = 75,0.5
CLASS_THRESH               = 0.80

@st.cache_data(ttl=600)
def load_hist(coin, vs):
    js = fetch_json(f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
                    {"vs_currency":vs,"days":H_DAYS})
    if not js or "prices" not in js:
        st.warning(f"⚠️ Failed to load {coin} history.")
        return pd.DataFrame(columns=["price","return"])
    df=pd.DataFrame(js["prices"], columns=["ts","price"])
    df["date"]=pd.to_datetime(df["ts"],unit="ms")
    df=df.set_index("date").resample("D").last().dropna()
    df["price"]=df["price"].astype(float)
    df["return"]=df["price"].pct_change(fill_method=None)*100
    return df

@st.cache_data(ttl=60)
def load_live():
    def one(cid,vs,extra):
        js=fetch_json("https://api.coingecko.com/api/v3/simple/price",
                      {"ids":cid,"vs_currencies":vs,**extra})
        return js or {}
    with concurrent.futures.ThreadPoolExecutor() as ex:
        b=ex.submit(one,"bitcoin","usd",{"include_24hr_change":"true"})
        x=ex.submit(one,"ripple","btc", {"include_24hr_change":"false"})
    j1,j2=b.result(),x.result()
    btc,xrp=j1.get("bitcoin",{}),j2.get("ripple",{})
    if not btc or not xrp:
        st.warning("⚠️ Failed to load live prices.")
    return {"BTC":(btc.get("usd",np.nan),btc.get("usd_24h_change",np.nan)),
            "XRP":(xrp.get("btc",np.nan),None)}

btc_usd=load_hist("bitcoin","usd")
xrp_usd=load_hist("ripple","usd")
live=load_live()
btc_p,btc_ch=live["BTC"]; xrp_p,_=live["XRP"]
# fallback to hist close if NaN
if np.isnan(btc_p) and not btc_usd.empty: btc_p=btc_usd["price"].iat[-1]
if np.isnan(xrp_p) and not xrp_usd.empty: xrp_p=xrp_usd["price"].iat[-1]

common=btc_usd.index.intersection(xrp_usd.index)
btc_usd=btc_usd.reindex(common); xrp_usd=xrp_usd.reindex(common)
xrp_btc=pd.DataFrame(index=common)
xrp_btc["price"]=xrp_usd["price"]/btc_usd["price"]
xrp_btc["return"]=xrp_btc["price"].pct_change()*100

def compute_ind(df):
    df["ema50"]=df["price"].ewm(span=EMA_T,adjust=False).mean()
    df["sma5"]=df["price"].rolling(5).mean()
    df["sma20"]=df["price"].rolling(20).mean()
    df["vol14"]=df["return"].rolling(VOL_W).std().fillna(0)
    d=df["price"].diff(); g=d.clip(lower=0).rolling(RSI_W).mean()
    l=-d.clip(upper=0).rolling(RSI_W).mean()
    df["rsi"]=100-100/(1+g/l.replace(0,np.nan))
    return df.dropna()

btc_hist=compute_ind(btc_usd.copy()); xrp_hist=compute_ind(xrp_btc.copy())

# ── ML & Signal Generation ──
btc_params=(75,1.5,1.0); xrp_params=(10,75,50,1.0)
def gen_sig(df,is_b,ps):
    X,y=[],[]
    for i in range(EMA_T,len(df)-1):
        p,ret,vol=df["price"].iat[i],df["return"].iat[i],df["vol14"].iat[i]
        ed=p-df["ema50"].iat[i]; mo=df["sma5"].iat[i]-df["sma20"].iat[i]; rs=df["rsi"].iat[i]
        if is_b: cond=ed>0 and mo>0 and rs<ps[0] and ret>=vol
        else:
            m,b,_,dip=ps; mv=df["price"].rolling(m).mean().iat[i]
            cond=p<mv and ((mv-p)/p*100)>=dip and vol>df["vol14"].iat[i-1]
        if not cond: continue
        X.append([rs,vol,ed,mo,ret]); y.append(1 if df["price"].iat[i+1]>p else 0)
    return np.array(X),np.array(y)

# (Synth & real-today samples appended as before...)
# Online SGDClassifier.partial_fit as before...

# ── Bot Logic & Rendering ──
def regime_ok(df,prob):
    rsi_b=BASE_RSI_OB+min(10,df["vol14"].iat[-1]*100)
    return {"Price>EMA50":df["price"].iat[-1]>df["ema50"].iat[-1],
            "SMA5>SMA20":df["sma5"].iat[-1]>df["sma20"].iat[-1],
            "RSI<Bound":df["rsi"].iat[-1]<rsi_b,
            "Vol≥Floor":df["vol14"].iat[-1]>=MIN_VOL,
            "ML Prob":prob>=CLASS_THRESH}

def compute_drop(df,pr,chg):
    vol=df["vol14"].iat[-1]; ret=chg if chg is not None else df["return"].iat[-1]
    if ret<vol: return None
    return vol if ret<=2*vol else 2*vol

def auto_state(key,hist,pr,chg,prob,low_c,up_c,cnt_c):
    bal=st.session_state.bal_b if key=="b" else st.session_state.bal_x
    dep=st.session_state[f"deployed_{key}"]; term=st.session_state[f"terminated_{key}"]
    drop= hist["vol14"].iat[-1] if (st.session_state.mode=="new" and not dep) else compute_drop(hist,pr,chg)
    if term and all(regime_ok(hist,prob).values()):
        st.session_state[f"terminated_{key}"]=False; term=False
    if not dep and not term and all(regime_ok(hist,prob).values()):
        st.session_state[f"deployed_{key}"]=True; dep=True
    low=pr*(1-drop/100) if (st.session_state.mode=="new" and drop) else low_c
    up=pr if st.session_state.mode=="new" else up_c
    sl=pr*(1-stop_loss/100); tp=up*(1+drop*1.5/100) if drop else up_c
    try:
        rec=max(5,min(30,int((bal/max(pr,1e-8))//((usd_tot/30)/max(pr,1e-8)))))
    except:
        rec=20
    grids=cnt_c if st.session_state.mode=="cont" else (manual_b if key=="b" and override else manual_x if key=="x" and override else rec)
    today=hist["price"].iat[-1]
    if not dep: act="Not Deployed"
    elif term: act="Terminated"
    elif today>=tp: act="Take-Profit"
    elif today<=sl: act="Stop-Loss"
    elif dep and drop: act="Redeploy"
    else: act="Hold"
    if compound and act in ("Take-Profit","Stop-Loss"):
        factor=(1+drop*1.5/100) if act=="Take-Profit" else (1-stop_loss/100)
        if key=="b": st.session_state.bal_b*=factor
        else:        st.session_state.bal_x*=factor
    return low,up,tp,sl,grids,rec,act

for key,label,hist,(pr,ch),prob in [
    ("b","🟡 BTC/USDT",btc_hist,(btc_p,btc_ch),p_b),
    ("x","🟣 XRP/BTC",  xrp_hist,(xrp_p,None),    p_x)
]:
    low,up,tp,sl,grids,rec,act=auto_state(key,hist,pr,ch,prob,
        st.session_state.get(f"cont_low_{key}",pr),
        st.session_state.get(f"cont_up_{key}",pr),
        st.session_state.get(f"cont_grids_{key}",30))
    st.subheader(f"{label} Bot")
    diag=regime_ok(hist,prob)
    with st.expander("🔧 Diagnostics",expanded=False):
        for k,v in diag.items(): st.write(f"{k}: {'✅' if v else '❌'}")
        st.write(f"ML Prob: {prob:.2f} ≥ {CLASS_THRESH}")
    if act=="Not Deployed":
        st.warning("⚠️ Waiting to deploy—adjust settings or override.")
        continue
    c1,c2=st.columns(2)
    if st.session_state.mode=="new": c1.metric("Grid Levels",grids); c2.metric("Recommended",rec)
    else:                          c1.metric("Grid Levels",grids); c2.write("")
    st.metric("Lower Price",f"{low:,.6f}")
    st.metric("Upper Price",f"{up:,.6f}")
    st.metric("Take-Profit At",f"{tp:,.6f}")
    st.metric("Stop-Loss At",f"{sl:,.6f}")
    if act=="Redeploy":     st.info("🔔 Redeploy signal")
    elif act=="Take-Profit":st.success("💰 TAKE-PROFIT")
    elif act=="Stop-Loss":  st.error("🔻 STOP-LOSS")
    elif act=="Terminated": st.error("🛑 TERMINATED")
    else:                   st.info("⏸ HOLD")

persist_all()
