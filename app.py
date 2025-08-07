# app.py
import os, shutil, pickle, time, random
from datetime import datetime
import pandas as pd, numpy as np
import streamlit as st
import requests
from sklearn.linear_model import SGDClassifier
from streamlit_autorefresh import st_autorefresh
import pytz

# ──────── Configuration ────────
STATE_DIR   = ".state"
ML_BUF_FILE = os.path.join(STATE_DIR, "ml_buffer.pkl")
FLAGS_FILE  = os.path.join(STATE_DIR, "flags.pkl")
MIN_VOL = 0.01

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
    with open(ML_BUF_FILE, "wb") as f:
        pickle.dump(buf, f)

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

def fetch_json(url):
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def load_live():
    data = {}
    for symbol in ["BTC", "XRP"]:
        resp = fetch_json(f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd")
        if resp and symbol.lower() in resp:
            data[symbol] = (resp[symbol.lower()]["usd"], datetime.utcnow())
        else:
            data[symbol] = (np.nan, None)
    return data

def today_feat(df):
    i = len(df) - 1
    if i < 0 or df.isna().iloc[i].any():
        return [[0, 0, 0, 0, 0]]
    return [[
        df["rsi"].iat[i],
        df["vol14"].iat[i],
        df["price"].iat[i] - df["ema50"].iat[i],
        df["sma5"].iat[i] - df["sma20"].iat[i],
        df["return"].iat[i],
    ]]

def compute_drop(df, pr, chg):
    vol = df["vol14"].iat[-1]
    ret = chg if chg is not None else df["return"].iat[-1]
    if pd.isna(vol) or pd.isna(ret) or vol <= 0:
        return None
    if ret < vol:
        return None
    return min(2 * vol, vol if ret <= 2 * vol else 2 * vol)

# ──────── Streamlit Setup ────────
st.set_page_config(layout="centered")
st_autorefresh(interval=60_000, key="refresh")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
st.caption(f"Last updated: {datetime.now(pytz.timezone('Europe/London')):%Y-%m-%d %H:%M %Z}")

# Load persisted state
flags = load_flags()
ml_buf = load_ml_buffer()
for k, v in flags.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("mem_X", ml_buf["X"])
st.session_state.setdefault("mem_y", ml_buf["y"])
st.session_state.setdefault("mem_ts", ml_buf["ts"])

if "online_clf" not in st.session_state:
    clf = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
    clf.partial_fit(np.zeros((2,5)), [0,1], classes=[0,1])
    st.session_state.online_clf = clf

# ──────── Sidebar: Persistence & Strategy ────────
st.sidebar.header("🛠️ Persistence & Resets")
if st.sidebar.button("Reset Bot State"):
    st.session_state.deployed_b = False
    st.session_state.terminated_b = False
    st.session_state.deployed_x = False
    st.session_state.terminated_x = False
if st.sidebar.button("Reset Balances"):
    st.session_state.bal_b = None
    st.session_state.bal_x = None
if st.sidebar.button("Clear ML Memory"):
    st.session_state.mem_X = []
    st.session_state.mem_y = []
    st.session_state.mem_ts = []
if st.sidebar.button("Delete Everything"):
    shutil.rmtree(STATE_DIR, ignore_errors=True)
    st.experimental_rerun()

st.sidebar.header("💰 Strategy Settings")
total_investment = st.sidebar.number_input("Total Investment ($)", value=3000.0, step=100.0)
btc_pct = st.sidebar.slider("BTC Allocation (%)", 0, 100, 70)
gbp_usd = st.sidebar.number_input("GBP/USD Rate", value=1.27, step=0.01)
btc_alloc = btc_pct / 100 * total_investment
xrp_alloc = total_investment - btc_alloc

st.sidebar.markdown(f"**Portfolio**")
st.sidebar.metric("Total", f"${total_investment:,.2f}", "£{:,.2f}".format(total_investment / gbp_usd))

stop_loss_pct = st.sidebar.slider("Stop-Loss (%)", 1.0, 5.0, 2.0, step=0.1)
compound = st.sidebar.toggle("📈 Enable Compounding")

# ──────── Live Data Display ────────
st.subheader("📊 Live Prices")
live = load_live()
cols = st.columns(2)
for i, symbol in enumerate(["BTC", "XRP"]):
    price, updated = live[symbol]
    if pd.isna(price):
        cols[i].error(f"{symbol}: Failed to fetch price")
    else:
        cols[i].metric(f"{symbol}/USD", f"${price:,.2f}", delta=None)

# NOTE: Add additional visual elements or strategy simulation below if desired.
