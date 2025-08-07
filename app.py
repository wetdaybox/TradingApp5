# app.py
import os
import shutil
import pickle
import time
import random
from datetime import datetime
import requests
import concurrent.futures

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import SGDClassifier
from streamlit_autorefresh import st_autorefresh
import pytz

# ─────────────── Persistence Helpers ───────────────
STATE_DIR   = ".state"
ML_BUF_FILE = os.path.join(STATE_DIR, "ml_buffer.pkl")
FLAGS_FILE  = os.path.join(STATE_DIR, "flags.pkl")

def ensure_state_dir():
    if not os.path.isdir(STATE_DIR):
        os.makedirs(STATE_DIR)

def load_flags():
    ensure_state_dir()
    if os.path.isfile(FLAGS_FILE):
        try:
            with open(FLAGS_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {
        "deployed_b": False, "terminated_b": False,
        "deployed_x": False, "terminated_x": False,
        "bal_b": None,        "bal_x": None
    }

def save_flags(flags):
    ensure_state_dir()
    with open(FLAGS_FILE, "wb") as f:
        pickle.dump(flags, f)

def load_ml_buffer():
    ensure_state_dir()
    if os.path.isfile(ML_BUF_FILE):
        try:
            with open(ML_BUF_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {"X": [], "y": [], "ts": []}

def save_ml_buffer(buf):
    ensure_state_dir()
    with open(ML_BUF_FILE, "wb") as f:
        pickle.dump(buf, f)

def persist_all():
    save_flags({
        "deployed_b": st.session_state.deployed_b,
        "terminated_b": st.session_state.terminated_b,
        "deployed_x": st.session_state.deployed_x,
        "terminated_x": st.session_state.terminated_x,
        "bal_b": st.session_state.bal_b,
        "bal_x": st.session_state.bal_x,
        "mode":    st.session_state.mode
    })
    save_ml_buffer({
        "X": st.session_state.mem_X,
        "y": st.session_state.mem_y,
        "ts": st.session_state.mem_ts
    })

# ───────── Streamlit Setup ─────────
st.set_page_config(layout="centered")
st_autorefresh(interval=60_000, key="refresh")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
st.caption(f"Last updated: {datetime.now(pytz.timezone('Europe/London')):%Y-%m-%d %H:%M %Z}")

# ───────── Restore State ─────────
flags  = load_flags()
ml_buf = load_ml_buffer()
for k, v in flags.items():
    st.session_state.setdefault(k, v)

# **Initialize mode if missing**
st.session_state.setdefault("mode", "new")

st.session_state.setdefault("mem_X", ml_buf["X"])
st.session_state.setdefault("mem_y", ml_buf["y"])
st.session_state.setdefault("mem_ts", ml_buf["ts"])

# online classifier
if "online_clf" not in st.session_state:
    clf = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
    # Bootstrap on 90-day real history later in code
    clf.partial_fit(np.zeros((2,5)), [0,1], classes=[0,1])
    st.session_state.online_clf = clf

# ───────── Persistence & Resets ─────────
st.sidebar.header("⚙️ Persistence & Resets")
if st.sidebar.button("🔄 Reset Bot State"):
    for b in ("b","x"):
        st.session_state[f"deployed_{b}"]   = False
        st.session_state[f"terminated_{b}"] = False
    st.sidebar.success("Bot state reset.")
if st.sidebar.button("🔄 Reset Balances"):
    st.session_state.bal_b = None
    st.session_state.bal_x = None
    st.sidebar.success("Balances reset.")
if st.sidebar.button("🔄 Clear ML Memory"):
    st.session_state.mem_X  = []
    st.session_state.mem_y  = []
    st.session_state.mem_ts = []
    st.sidebar.success("ML memory cleared.")
if st.sidebar.button("🗑️ Delete Everything"):
    if os.path.isdir(STATE_DIR):
        shutil.rmtree(STATE_DIR)
    for k in ("deployed_b","terminated_b","deployed_x","terminated_x",
              "bal_b","bal_x","mem_X","mem_y","mem_ts","mode"):
        st.session_state.pop(k, None)
    st.sidebar.success("All state deleted.")
    try:
        st.experimental_rerun()
    except AttributeError:
        st.stop()

# ───────── Strategy Settings ─────────
st.sidebar.header("💰 Strategy Settings")
usd_tot   = st.sidebar.number_input("Total Investment ($)",100.0,1e6,3000.0,100.0)
pct_btc   = st.sidebar.slider("BTC Allocation (%)",0,100,70)
usd_btc   = usd_tot * pct_btc/100
usd_xrp   = usd_tot - usd_btc
gbp_rate  = st.sidebar.number_input("GBP/USD Rate",1.10,1.60,1.27,0.01)
st.sidebar.metric("Portfolio",f"${usd_tot:,.2f}",f"£{usd_tot/gbp_rate:,.2f}")
stop_loss = st.sidebar.slider("Stop-Loss (%)",1.0,5.0,2.0,0.1)
compound  = st.sidebar.checkbox("Enable Compounding", value=False)
mode_sel  = st.sidebar.radio("Mode",["Start New Cycle","Continue Existing"],
                             index=0 if st.session_state.mode=="new" else 1)
st.session_state.mode = "new" if mode_sel=="Start New Cycle" else "cont"
override  = st.sidebar.checkbox("Manual Grid Override", value=False)
manual_b  = st.sidebar.number_input("BTC Grids",2,30,6) if (override and st.session_state.mode=="new") else None
manual_x  = st.sidebar.number_input("XRP Grids",2,30,8) if (override and st.session_state.mode=="new") else None

if st.session_state.bal_b is None:
    st.session_state.bal_b = usd_btc
if st.session_state.bal_x is None:
    st.session_state.bal_x = usd_xrp

# ───────── Data & Indicators ─────────
H_DAYS, VOL_W, RSI_W, EMA_T = 90,14,14,50
BASE_RSI_OB, MIN_VOL        = 75,0.5
CLASS_THRESH, MAX_RETRIES   = 0.80, 3

def fetch_json(url, params=None):
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 429:
                time.sleep(2**i)
                continue
            r.raise_for_status()
            return r.json()
        except:
            time.sleep(2**i)
    return None

@st.cache_data(ttl=600)
def load_hist(coin, vs):
    js = fetch_json(f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
                    {"vs_currency":vs,"days":H_DAYS})
    if not js or "prices" not in js:
        return pd.DataFrame()
    df = pd.DataFrame(js["prices"], columns=["ts","price"])
    df["date"] = pd.to_datetime(df["ts"],unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["price"]  = df["price"].astype(float)
    df["return"] = df["price"].pct_change()*100
    return df

def load_live():
    def one(cid,vs,extra):
        j = fetch_json("https://api.coingecko.com/api/v3/simple/price",
                       {"ids":cid,"vs_currencies":vs,**extra})
        return j or {}
    try:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            b = ex.submit(one,"bitcoin","usd",{"include_24hr_change":"true"})
            x = ex.submit(one,"ripple","btc",{"include_24hr_change":"false"})
        j1,j2 = b.result(), x.result()
        return {
            "BTC": (j1.get("bitcoin",{}).get("usd",np.nan), j1.get("bitcoin",{}).get("usd_24h_change",np.nan)),
            "XRP": (j2.get("ripple",{}).get("btc",np.nan), None)
        }
    except:
        return {"BTC":(np.nan,None),"XRP":(np.nan,None)}

def compute_ind(df):
    df["ema50"] = df["price"].ewm(span=EMA_T,adjust=False).mean()
    df["sma5"]  = df["price"].rolling(5).mean()
    df["sma20"] = df["price"].rolling(20).mean()
    df["vol14"] = df["return"].rolling(VOL_W).std().fillna(0)
    d = df["price"].diff()
    g = d.clip(lower=0).rolling(RSI_W).mean()
    l = -d.clip(upper=0).rolling(RSI_W).mean()
    df["rsi"] = 100 - 100/(1 + g/l.replace(0,np.nan))
    return df.dropna()

# fetch and prepare
btc_usd = load_hist("bitcoin","usd")
xrp_usd = load_hist("ripple","usd")
if btc_usd.empty or xrp_usd.empty:
    st.error("❌ Failed to load history."); st.stop()
idx       = btc_usd.index.intersection(xrp_usd.index)
btc_usd   = btc_usd.reindex(idx); xrp_usd = xrp_usd.reindex(idx)
xrp_btc   = pd.DataFrame({"price":xrp_usd["price"]/btc_usd["price"]}, index=idx)
xrp_btc["return"] = xrp_btc["price"].pct_change()*100
btc_hist  = compute_ind(btc_usd.copy())
xrp_hist  = compute_ind(xrp_btc.copy())

# bootstrap classifier on real features
if st.session_state.online_clf.coef_.shape == (1,5):
    Xb, yb = gen_sig(btc_hist,True,(75,1.5,1.0))
    Xx, yx = gen_sig(xrp_hist,False,(10,75,50,1.0))
    X0 = np.vstack([Xb,Xx]) if len(Xb) and len(Xx) else np.zeros((2,5))
    y0 = np.concatenate([yb,yx]) if len(yb) and len(yx) else np.array([0,1])
    st.session_state.online_clf.partial_fit(X0, y0, classes=[0,1])

# live prediction
live = load_live()
btc_p, btc_ch = live["BTC"]; xrp_p,_ = live["XRP"]
if np.isnan(btc_p) or np.isnan(xrp_p):
    st.error("❌ Failed to load live data."); st.stop()

# proceed with original scenario augmentation, auto_state, rendering...
# (rest of your code follows unchanged, using st.session_state.mode safely)
