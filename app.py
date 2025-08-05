# app.py
import os, shutil, pickle, time, random
from datetime import datetime
import requests, concurrent.futures
import pandas as pd, numpy as np
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
        return pickle.load(open(FLAGS_FILE, "rb"))
    return {
        "deployed_b": False, "terminated_b": False,
        "deployed_x": False, "terminated_x": False,
        "bal_b": None,        "bal_x": None
    }

def save_flags(f):
    pickle.dump(f, open(FLAGS_FILE, "wb"))

def load_ml_buffer():
    ensure_state_dir()
    if os.path.isfile(ML_BUF_FILE):
        return pickle.load(open(ML_BUF_FILE, "rb"))
    return {"X": [], "y": [], "ts": []}

def save_ml_buffer(buf):
    pickle.dump(buf, open(ML_BUF_FILE, "wb"))

def persist_all():
    save_flags({
        "deployed_b": st.session_state.deployed_b,
        "terminated_b": st.session_state.terminated_b,
        "deployed_x": st.session_state.deployed_x,
        "terminated_x": st.session_state.terminated_x,
        "bal_b": st.session_state.bal_b,
        "bal_x": st.session_state.bal_x
    })
    save_ml_buffer({
        "X": st.session_state.mem_X,
        "y": st.session_state.mem_y,
        "ts": st.session_state.mem_ts
    })

# restore persisted state
flags  = load_flags()
ml_buf = load_ml_buffer()

# ────────────── Streamlit Setup ──────────────
st.set_page_config(layout="centered")
st_autorefresh(interval=60_000, key="refresh")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
st.caption(f"Last updated: {datetime.now(pytz.timezone('Europe/London')):%Y-%m-%d %H:%M %Z}")

# initialize session_state
for k,v in flags.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("mem_X", ml_buf["X"])
st.session_state.setdefault("mem_y", ml_buf["y"])
st.session_state.setdefault("mem_ts", ml_buf["ts"])

# online learner
if "online_clf" not in st.session_state:
    clf = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
    clf.partial_fit(np.zeros((2,5)), [0,1], classes=[0,1])
    st.session_state.online_clf = clf

# ────────────── Robust fetch_json ──────────────
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

# ────────────── Data Loading ──────────────
H_DAYS = 90
@st.cache_data(ttl=600)
def load_hist(coin, vs):
    js = fetch_json(f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
                    {"vs_currency": vs, "days": H_DAYS})
    if not js or "prices" not in js:
        st.warning(f"⚠️ Failed to load {coin} history.")
        return pd.DataFrame(columns=["price","return"])
    df = pd.DataFrame(js["prices"], columns=["ts","price"])
    df["date"]   = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["price"]  = df["price"].astype(float)
    df["return"] = df["price"].pct_change()*100
    return df

@st.cache_data(ttl=60)
def load_live():
    def one(cid, vs, extra):
        js = fetch_json("https://api.coingecko.com/api/v3/simple/price",
                        {"ids":cid,"vs_currencies":vs,**extra})
        return js or {}
    with concurrent.futures.ThreadPoolExecutor() as ex:
        f1 = ex.submit(one, "bitcoin", "usd", {"include_24hr_change":"true"})
        f2 = ex.submit(one, "ripple", "btc", {"include_24hr_change":"false"})
    j1, j2 = f1.result(), f2.result()
    btc = j1.get("bitcoin", {})
    xrp = j2.get("ripple", {})
    if not btc or not xrp:
        st.warning("⚠️ Failed to load live prices.")
    return {
        "BTC": (btc.get("usd", np.nan), btc.get("usd_24h_change", np.nan)),
        "XRP": (xrp.get("btc", np.nan), None)
    }

btc_usd = load_hist("bitcoin","usd")
xrp_usd = load_hist("ripple","usd")
# build ratio series
common = btc_usd.index.intersection(xrp_usd.index)
btc_usd = btc_usd.reindex(common)
xrp_usd = xrp_usd.reindex(common)
xrp_btc = pd.DataFrame(index=common)
xrp_btc["price"]  = xrp_usd["price"] / btc_usd["price"]
xrp_btc["return"] = xrp_btc["price"].pct_change()*100

# indicators
def compute_indicators(df):
    df["ema50"] = df["price"].ewm(span=50, adjust=False).mean()
    df["sma5"]  = df["price"].rolling(5).mean()
    df["sma20"] = df["price"].rolling(20).mean()
    df["vol14"] = df["return"].rolling(14).std().fillna(0)
    delta = df["price"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = -delta.clip(upper=0).rolling(14).mean()
    df["rsi"]  = 100 - 100/(1 + gain/loss.replace(0, np.nan))
    return df.dropna()

btc_hist = compute_indicators(btc_usd.copy())
xrp_hist = compute_indicators(xrp_btc.copy())

live       = load_live()
btc_p,btc_ch = live["BTC"]
xrp_p,_      = live["XRP"]

# ────────────── Sidebar: Settings & Resets ──────────────
st.sidebar.header("💰 Strategy Settings")
usd_tot   = st.sidebar.number_input("Total Investment ($)", 100.0, 1e6, 3000.0, 100.0)
pct_btc   = st.sidebar.slider("BTC Allocation (%)", 0, 100, 70)
usd_btc   = usd_tot * pct_btc/100
usd_xrp   = usd_tot - usd_btc
gbp_rate  = st.sidebar.number_input("GBP/USD Rate",1.10,1.60,1.27,0.01)
st.sidebar.metric("Portfolio", f"${usd_tot:,.2f}", f"£{usd_tot/gbp_rate:,.2f}")

stop_loss = st.sidebar.slider("Stop-Loss (%)",1.0,5.0,2.0,0.1)
compound  = st.sidebar.checkbox("Enable Compounding", value=False)

mode = st.sidebar.radio("Mode", ["Start New Cycle","Continue Existing"],
                       index=0 if st.session_state.get("mode","new")=="new" else 1)
st.session_state.mode = "new" if mode=="Start New Cycle" else "cont"

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
    shutil.rmtree(STATE_DIR, ignore_errors=True)
    for k in ("deployed_b","terminated_b","deployed_x","terminated_x",
              "bal_b","bal_x","mem_X","mem_y","mem_ts"):
        st.session_state.pop(k, None)
    st.sidebar.success("All state deleted.")
    st.experimental_rerun()

# initialize balances
if st.session_state.bal_b is None: st.session_state.bal_b = usd_btc
if st.session_state.bal_x is None: st.session_state.bal_x = usd_xrp

# ────────────── ML, Signals & Render ──────────────
# ... the unchanged ML generation, auto_state, and rendering code goes here ...
# ensure that after this block you call persist_all()

persist_all()
