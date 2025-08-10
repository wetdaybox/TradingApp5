# app.py
"""
Infinite Scalping Grid Bot Trading System
UI tweaks: compact grid display, full-price precision in main view,
and slightly improved diagnostics formatting. No logic changes.
"""

import os
import shutil
import pickle
import time
import random
from datetime import datetime
import requests
import concurrent.futures
import smtplib
from email.mime.text import MIMEText

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import SGDClassifier
from streamlit_autorefresh import st_autorefresh
import pytz

# ---------------- Paths & Constants ----------------
STATE_DIR   = ".state"
ML_BUF_FILE = os.path.join(STATE_DIR, "ml_buffer.pkl")
FLAGS_FILE  = os.path.join(STATE_DIR, "flags.pkl")

H_DAYS, VOL_W, RSI_W, EMA_T = 90, 14, 14, 50
BASE_RSI_OB, MIN_VOL        = 75, 0.5
CLASS_THRESH                = 0.80
MAX_RETRIES                 = 5

# ---------------- Persistence helpers ----------------
def ensure_state_dir():
    if not os.path.isdir(STATE_DIR):
        os.makedirs(STATE_DIR, exist_ok=True)

def load_pickle(path, default):
    ensure_state_dir()
    if os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return default
    return default

def save_pickle(obj, path):
    ensure_state_dir()
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_flags():
    return load_pickle(FLAGS_FILE, {
        "deployed_b": False, "terminated_b": False,
        "deployed_x": False, "terminated_x": False,
        "bal_b": None, "bal_x": None,
        "mode": "new"
    })

def save_flags(flags):
    save_pickle(flags, FLAGS_FILE)

def load_ml_buffer():
    return load_pickle(ML_BUF_FILE, {"X": [], "y": [], "ts": []})

def save_ml_buffer(buf):
    save_pickle(buf, ML_BUF_FILE)

def persist_all():
    save_flags({
        "deployed_b": st.session_state.deployed_b,
        "terminated_b": st.session_state.terminated_b,
        "deployed_x": st.session_state.deployed_x,
        "terminated_x": st.session_state.terminated_x,
        "bal_b": st.session_state.bal_b,
        "bal_x": st.session_state.bal_x,
        "mode": st.session_state.mode
    })
    save_ml_buffer({
        "X": st.session_state.mem_X,
        "y": st.session_state.mem_y,
        "ts": st.session_state.mem_ts
    })

# ---------------- Networking helpers ----------------
def fetch_json(url, params=None, headers=None):
    headers = headers or {"User-Agent": "Mozilla/5.0 (compatible; GridBot/1.0)"}
    backoff_base = 1.5
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=10, headers=headers)
            if r.status_code == 429:
                time.sleep((backoff_base ** i) + random.random())
                continue
            if 500 <= r.status_code < 600:
                time.sleep((backoff_base ** i) + random.random())
                continue
            r.raise_for_status()
            try:
                return r.json()
            except ValueError:
                return {"__raw_text__": r.text, "__status__": r.status_code}
        except requests.RequestException:
            time.sleep((backoff_base ** i) + random.random())
            continue
    return None

@st.cache_data(ttl=600)
def load_hist(coin, vs):
    now_s = int(time.time())
    from_s = now_s - H_DAYS * 86400
    url_range = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range"
    js = fetch_json(url_range, {"vs_currency": vs, "from": from_s, "to": now_s})
    prices = None
    if js and isinstance(js, dict) and js.get("prices"):
        prices = js["prices"]
    else:
        url_mc = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
        js2 = fetch_json(url_mc, {"vs_currency": vs, "days": H_DAYS})
        if js2 and isinstance(js2, dict) and js2.get("prices"):
            prices = js2["prices"]
    if not prices:
        return pd.DataFrame()
    try:
        df = pd.DataFrame(prices, columns=["ts", "price"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("date").resample("D").last().dropna()
        df["price"] = df["price"].astype(float)
        df["return"] = df["price"].pct_change() * 100
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_live():
    errors = []
    try:
        j1 = fetch_json("https://api.coingecko.com/api/v3/simple/price",
                        {"ids": "bitcoin", "vs_currencies": "usd", "include_24hr_change": "true"})
        j2 = fetch_json("https://api.coingecko.com/api/v3/simple/price",
                        {"ids": "ripple", "vs_currencies": "btc", "include_24hr_change": "false"})
        if j1 and isinstance(j1, dict) and "bitcoin" in j1 and "usd" in j1["bitcoin"]:
            btc_price = j1["bitcoin"].get("usd", np.nan)
            btc_change = j1["bitcoin"].get("usd_24h_change", None)
            xrp_price = None
            if j2 and isinstance(j2, dict) and "ripple" in j2 and "btc" in j2["ripple"]:
                xrp_price = j2["ripple"].get("btc", np.nan)
            if btc_price is not None and xrp_price is not None:
                return {"BTC": (float(btc_price), float(btc_change) if btc_change is not None else None),
                        "XRP": (float(xrp_price), None),
                        "source": "coingecko",
                        "errors": errors}
            else:
                errors.append("CoinGecko returned incomplete fields.")
        else:
            errors.append("CoinGecko basic request failed or returned no data.")
    except Exception as e:
        errors.append(f"CoinGecko exception: {e}")

    try:
        jb = fetch_json("https://api.binance.com/api/v3/ticker/24hr", {"symbol": "BTCUSDT"})
        if jb and isinstance(jb, dict) and "lastPrice" in jb:
            btc_price = float(jb["lastPrice"])
            btc_pct = None
            if "priceChangePercent" in jb:
                try:
                    btc_pct = float(jb["priceChangePercent"])
                except Exception:
                    btc_pct = None
        else:
            btc_price = np.nan
            btc_pct = None
            errors.append("Binance BTC ticker failed or returned unexpected structure.")

        jx = fetch_json("https://api.binance.com/api/v3/ticker/24hr", {"symbol": "XRPBTC"})
        if jx and isinstance(jx, dict) and "lastPrice" in jx:
            xrp_price = float(jx["lastPrice"])
        else:
            xrp_price = np.nan
            errors.append("Binance XRPBTC ticker failed or returned unexpected structure.")

        if (not np.isnan(btc_price)) and (not np.isnan(xrp_price)):
            return {"BTC": (btc_price, btc_pct),
                    "XRP": (xrp_price, None),
                    "source": "binance",
                    "errors": errors}
    except Exception as e:
        errors.append(f"Binance exception: {e}")

    return {"BTC": (np.nan, None), "XRP": (np.nan, None), "source": "none", "errors": errors}

# ---------------- Indicators & Feature Builders ----------------
def compute_ind(df):
    df["ema50"] = df["price"].ewm(span=EMA_T, adjust=False).mean()
    df["sma5"]  = df["price"].rolling(5).mean()
    df["sma20"] = df["price"].rolling(20).mean()
    df["vol14"] = df["return"].rolling(VOL_W).std().fillna(0)
    d = df["price"].diff()
    g = d.clip(lower=0).rolling(RSI_W).mean()
    l = -d.clip(upper=0).rolling(RSI_W).mean()
    df["rsi"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
    return df.dropna()

btc_params = (75, 1.5, 1.0)
xrp_params = (10, 75, 50, 1.0)

def gen_sig(df, is_b, params):
    X, y = [], []
    for i in range(EMA_T, len(df) - 1):
        p = df["price"].iat[i]
        ret = df["return"].iat[i]
        vol = df["vol14"].iat[i]
        ed = p - df["ema50"].iat[i]
        mo = df["sma5"].iat[i] - df["sma20"].iat[i]
        rs = df["rsi"].iat[i]
        if is_b:
            cond = ed > 0 and mo > 0 and rs < params[0] and ret >= vol
        else:
            m, b_, _, dip = params
            mv = df["price"].rolling(m).mean().iat[i]
            cond = p < mv and ((mv - p) / p * 100) >= dip and vol > df["vol14"].iat[i - 1]
        if not cond:
            continue
        X.append([rs, vol, ed, mo, ret])
        y.append(1 if df["price"].iat[i + 1] > p else 0)
    return np.array(X), np.array(y)

def generate_scenario(vol, reg, days=90):
    mapping = {
        "normal": (0, vol, None),
        "high-vol": (0, vol * 2, None),
        "crash": (-0.002, vol * 3, (-0.3,)),
        "rally": (0.002, vol * 1.5, (0.3,)),
        "flash-crash": (0, vol, (-0.5,))
    }
    μ, σ, jumps = mapping[reg]
    rets = np.random.normal(μ, σ, days)
    if jumps:
        for j in jumps:
            rets[random.randrange(days)] += j
    return 100 * np.cumprod(1 + rets)

def extract_Xy(prices, is_b):
    df2 = pd.DataFrame({"price": prices})
    df2["return"] = df2["price"].pct_change() * 100
    df2 = df2.dropna()
    if df2.empty:
        return np.array([]), np.array([])
    try:
        df2 = compute_ind(df2)
    except Exception:
        return np.array([]), np.array([])
    return gen_sig(df2, is_b, btc_params if is_b else xrp_params)

def today_feat(df):
    if len(df) == 0:
        return [[0, 0, 0, 0, 0]]
    i = len(df) - 1
    return [[
        df["rsi"].iat[i],
        df["vol14"].iat[i],
        df["price"].iat[i] - df["ema50"].iat[i],
        df["sma5"].iat[i] - df["sma20"].iat[i],
        df["return"].iat[i],
    ]]

# ---------------- UI Setup & State ----------------
st.set_page_config(layout="centered")
st_autorefresh(interval=60_000, key="refresh")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
st.caption(f"Last updated: {datetime.now(pytz.timezone('Europe/London')):%Y-%m-%d %H:%M %Z}")

flags = load_flags()
ml_buf = load_ml_buffer()
for k, v in flags.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("mem_X", ml_buf.get("X", []))
st.session_state.setdefault("mem_y", ml_buf.get("y", []))
st.session_state.setdefault("mem_ts", ml_buf.get("ts", []))
st.session_state.setdefault("mode", flags.get("mode", "new"))

# bootstrap minimal classifier placeholder
if "online_clf" not in st.session_state:
    clf = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
    clf.partial_fit(np.zeros((2,5)), [0,1], classes=[0,1])
    st.session_state.online_clf = clf

# ---------------- Sidebar (compact) ----------------
st.sidebar.header("📣 Alerts & Notifications")
st.session_state.email_alerts = st.sidebar.checkbox("Enable Email Alerts", value=False)
st.session_state.email_addr = st.sidebar.text_input("Yahoo Email Address", value="")
st.session_state.email_pass = st.sidebar.text_input("Yahoo App Password", type="password")

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
    except Exception:
        st.stop()

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
                             index=0 if st.session_state.get("mode","new")=="new" else 1)
st.session_state.mode = "new" if mode_sel=="Start New Cycle" else "cont"

if st.session_state.mode == "cont":
    st.sidebar.mark
