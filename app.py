# app.py
"""
Infinite Scalping Grid Bot Trading System
UI change: ALWAYS hide full grid levels by default.
Show a compact sample; full list inside a collapsed expander.
No changes to logic, ML, persistence, or alerts.
"""

import os, shutil, pickle, time, random
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

# ---------------- Config & Paths ----------------
STATE_DIR   = ".state"
ML_BUF_FILE = os.path.join(STATE_DIR, "ml_buffer.pkl")
FLAGS_FILE  = os.path.join(STATE_DIR, "flags.pkl")

H_DAYS, VOL_W, RSI_W, EMA_T = 90, 14, 14, 50
BASE_RSI_OB, MIN_VOL        = 75, 0.5
CLASS_THRESH                = 0.80
MAX_RETRIES                 = 4

# ---------------- Persistence ----------------
def ensure_state_dir():
    if not os.path.isdir(STATE_DIR):
        os.makedirs(STATE_DIR, exist_ok=True)

def load_pickle(path, default):
    ensure_state_dir()
    try:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return default

def save_pickle(obj, path):
    ensure_state_dir()
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_flags():
    return load_pickle(FLAGS_FILE, {
        "deployed_b": False, "terminated_b": False,
        "deployed_x": False, "terminated_x": False,
        "bal_b": None, "bal_x": None, "mode": "new"
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
    headers = headers or {"User-Agent": "GridBot/1.0"}
    backoff = 1.5
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=10, headers=headers)
            if r.status_code == 429:
                time.sleep(backoff ** i + random.random())
                continue
            if 500 <= r.status_code < 600:
                time.sleep(backoff ** i + random.random())
                continue
            r.raise_for_status()
            try:
                return r.json()
            except ValueError:
                return {"__raw_text__": r.text, "__status__": r.status_code}
        except requests.RequestException:
            time.sleep(backoff ** i + random.random())
            continue
    return None

@st.cache_data(ttl=600)
def load_hist(coin_id, vs_currency):
    """
    Return pd.DataFrame with index=date and columns: price, return
    Tries CoinGecko (range), falls back to CoinGecko market_chart, then Binance klines.
    """
    errors = []
    now_s = int(time.time())
    from_s = now_s - H_DAYS * 86400

    # 1) CoinGecko range
    url_range = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    j = fetch_json(url_range, {"vs_currency": vs_currency, "from": from_s, "to": now_s})
    if j and isinstance(j, dict) and j.get("prices"):
        prices = j["prices"]
    else:
        # 2) CoinGecko market_chart (days)
        url_mc = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        j2 = fetch_json(url_mc, {"vs_currency": vs_currency, "days": H_DAYS})
        if j2 and isinstance(j2, dict) and j2.get("prices"):
            prices = j2["prices"]
        else:
            prices = None
            errors.append("CoinGecko history failed")

    # 3) Binance klines fallback (only for common pairs like BTCUSDT / XRPUSDT)
    if not prices:
        try:
            mapping = {"bitcoin": "BTCUSDT", "ripple": "XRPUSDT"}
            symbol = mapping.get(coin_id)
            if symbol:
                kb = fetch_json("https://api.binance.com/api/v3/klines", {"symbol": symbol, "interval": "1d", "limit": H_DAYS})
                if kb and isinstance(kb, list):
                    prices = [[int(k[0]), float(k[4])] for k in kb]
                else:
                    errors.append("Binance klines failed or empty")
        except Exception as e:
            errors.append(f"Binance klines exception: {e}")

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
    # Try CoinGecko
    try:
        j1 = fetch_json("https://api.coingecko.com/api/v3/simple/price",
                        {"ids": "bitcoin", "vs_currencies": "usd", "include_24hr_change": "true"})
        j2 = fetch_json("https://api.coingecko.com/api/v3/simple/price",
                        {"ids": "ripple", "vs_currencies": "btc"})
        if j1 and isinstance(j1, dict) and "bitcoin" in j1 and "usd" in j1["bitcoin"]:
            btc_price = float(j1["bitcoin"].get("usd", np.nan))
            btc_ch = j1["bitcoin"].get("usd_24h_change", None)
            xrp_price = float(j2["ripple"].get("btc", np.nan)) if (j2 and "ripple" in j2) else np.nan
            if not np.isnan(btc_price) and not np.isnan(xrp_price):
                return {"BTC": (btc_price, btc_ch), "XRP": (xrp_price, None), "source": "coingecko", "errors": errors}
        errors.append("CoinGecko live incomplete")
    except Exception as e:
        errors.append(f"CoinGecko live err: {e}")

    # Try Binance
    try:
        jb = fetch_json("https://api.binance.com/api/v3/ticker/24hr", {"symbol": "BTCUSDT"})
        jx = fetch_json("https://api.binance.com/api/v3/ticker/24hr", {"symbol": "XRPBTC"})
        btc_price = float(jb["lastPrice"]) if (jb and "lastPrice" in jb) else np.nan
        btc_ch = float(jb["priceChangePercent"]) if (jb and "priceChangePercent" in jb) else None
        xrp_price = float(jx["lastPrice"]) if (jx and "lastPrice" in jx) else np.nan
        if not np.isnan(btc_price) and not np.isnan(xrp_price):
            return {"BTC": (btc_price, btc_ch), "XRP": (xrp_price, None), "source": "binance", "errors": errors}
        errors.append("Binance live incomplete")
    except Exception as e:
        errors.append(f"Binance live err: {e}")

    return {"BTC": (np.nan, None), "XRP": (np.nan, None), "source": "none", "errors": errors}

# ---------------- Indicators, ML, scenarios ----------------
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
    X,y = [],[]
    for i in range(EMA_T, len(df)-1):
        p = df["price"].iat[i]
        ret = df["return"].iat[i]
        vol = df["vol14"].iat[i]
        ed = p - df["ema50"].iat[i]
        mo = df["sma5"].iat[i] - df["sma20"].iat[i]
        rs = df["rsi"].iat[i]
        if is_b:
            cond = ed>0 and mo>0 and rs<params[0] and ret>=vol
        else:
            m,b_,_,dip = params
            mv = df["price"].rolling(m).mean().iat[i]
            cond = p < mv and ((mv-p)/p*100)>=dip and vol>df["vol14"].iat[i-1]
        if not cond: continue
        X.append([rs,vol,ed,mo,ret])
        y.append(1 if df["price"].iat[i+1]>p else 0)
    return np.array(X), np.array(y)

def generate_scenario(vol,reg,days=90):
    mapping = {
        "normal":      (0,vol,None),
        "high-vol":    (0,vol*2,None),
        "crash":       (-0.002,vol*3,(-0.3,)),
        "rally":       (0.002,vol*1.5,(0.3,)),
        "flash-crash": (0,vol,(-0.5,))
    }
    μ,σ,jumps = mapping[reg]
    rets = np.random.normal(μ,σ,days)
    if jumps:
        for j in jumps: rets[random.randrange(days)] += j
    return 100 * np.cumprod(1+rets)

def extract_Xy(prices,is_b):
    df = pd.DataFrame({"price":prices})
    df["return"] = df["price"].pct_change()*100
    df = df.dropna()
    if df.empty: return np.array([]), np.array([])
    try:
        df = compute_ind(df)
    except Exception:
        return np.array([]), np.array([])
    return gen_sig(df,is_b, btc_params if is_b else xrp_params)

# ---------------- UI & state init ----------------
st.set_page_config(layout="centered")
st_autorefresh(interval=60_000, key="refresh")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
st.caption(f"Last updated: {datetime.now(pytz.timezone('Europe/London')):%Y-%m-%d %H:%M %Z}")

flags = load_flags()
ml_buf = load_ml_buffer()

defaults = {
    **flags,
    "mem_X": ml_buf.get("X",[]),
    "mem_y": ml_buf.get("y",[]),
    "mem_ts": ml_buf.get("ts",[]),
    "online_clf": None,
    "email_alerts": False,
    "email_addr": "",
    "email_pass": "",
    "spacing_type": "Arithmetic"
}
for k,v in defaults.items():
    st.session_state.setdefault(k,v)

for f in ("deployed_b","terminated_b","deployed_x","terminated_x"):
    st.session_state.setdefault(f, False)

if st.session_state.online_clf is None:
    clf = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
    try:
        clf.partial_fit(np.zeros((2,5)), [0,1], classes=[0,1])
    except Exception:
        pass
    st.session_state.online_clf = clf

# ---------------- Sidebar controls ----------------
st.sidebar.header("📣 Alerts & Notifications")
st.session_state.email_alerts = st.sidebar.checkbox("Enable Email Alerts", value=st.session_state.email_alerts)
st.session_state.email_addr = st.sidebar.text_input("Yahoo Email Address", value=st.session_state.email_addr)
st.session_state.email_pass = st.sidebar.text_input("Yahoo App Password", type="password", value=st.session_state.email_pass)

st.sidebar.header("⚙️ Persistence & Resets")
if st.sidebar.button("🔄 Reset Bot State"):
    for b in ("b","x"):
        st.session_state[f"deployed_{b}"] = False
        st.session_state[f"terminated_{b}"] = False
    st.sidebar.success("Bot state reset.")
if st.sidebar.button("🔄 Reset Balances"):
    st.session_state.bal_b = None
    st.session_state.bal_x = None
    st.sidebar.success("Balances reset.")
if st.sidebar.button("🔄 Clear ML Memory"):
    st.session_state.mem_X = []
    st.session_state.mem_y = []
    st.session_state.mem_ts = []
    st.sidebar.success("ML memory cleared.")
if st.sidebar.button("🗑️ Delete Everything"):
    if os.path.isdir(STATE_DIR):
        shutil.rmtree(STATE_DIR)
    for k in list(st.session_state.keys()):
        st.session_state.pop(k, None)
    st.sidebar.success("All state deleted.")

st.sidebar.header("💰 Strategy Settings")
usd_tot = st.sidebar.number_input("Total Investment ($)",100.0,1e7,3000.0,100.0)
pct_btc = st.sidebar.slider("BTC Allocation (%)",0,100,70)
usd_btc = usd_tot * pct_btc/100
usd_xrp = usd_tot - usd_btc
gbp_rate = st.sidebar.number_input("GBP/USD Rate",1.10,1.60,1.27,0.01)
st.sidebar.metric("Portfolio", f"${usd_tot:,.2f}", f"£{usd_tot/gbp_rate:,.2f}")
stop_loss = st.sidebar.slider("Stop-Loss (%)",1.0,10.0,2.0,0.1)
compound = st.sidebar.checkbox("Enable Compounding", value=False)
mode_sel = st.sidebar.radio("Mode", ["Start New Cycle","Continue Existing"],
                             index=0 if st.session_state.get("mode","new")=="new" else 1)
st.session_state.mode = "new" if mode_sel=="Start New Cycle" else "cont"

st.session_state.spacing_type = st.sidebar.radio("Grid Spacing", ["Arithmetic","Geometric"], index=0 if st.session_state.spacing_type=="Arithmetic" else 1)
override = st.sidebar.checkbox("Manual Grid Override", value=False)
manual_b = st.sidebar.number_input("BTC Grids (manual)",2,200,6) if (override and st.session_state.mode=="new") else None
manual_x = st.sidebar.number_input("XRP Grids (manual)",2,200,8) if (override and st.session_state.mode=="new") else None

if st.session_state.get("bal_b") is None: st.session_state.bal_b = usd_btc
if st.session_state.get("bal_x") is None: st.session_state.bal_x = usd_xrp

# ---------------- Load data & diagnostics ----------------
diag_msgs = []

btc_usd = load_hist("bitcoin","usd")
if btc_usd.empty:
    diag_msgs.append("History: CoinGecko+Binance failed for BTC 90-day.")

xrp_usd = load_hist("ripple","usd")
if xrp_usd.empty:
    diag_msgs.append("History: CoinGecko+Binance failed for XRP 90-day.")

if btc_usd.empty or xrp_usd.empty:
    diag_msgs.append("One or more 90-day histories are missing. Creating a small synthetic fallback so UI continues.")
    days = 90
    tindex = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D")
    btc_prices = np.linspace(100000.0, 110000.0, days) + np.random.normal(0, 100, days)
    xrp_usd_prices = np.linspace(0.5, 0.7, days) + np.random.normal(0, 0.01, days)
    btc_usd = pd.DataFrame({"price": btc_prices}, index=tindex)
    btc_usd["return"] = btc_usd["price"].pct_change()*100
    xrp_usd = pd.DataFrame({"price": xrp_usd_prices}, index=tindex)
    xrp_usd["return"] = xrp_usd["price"].pct_change()*100

idx = btc_usd.index.intersection(xrp_usd.index)
btc_usd = btc_usd.reindex(idx).dropna()
xrp_usd = xrp_usd.reindex(idx).dropna()
xrp_btc = pd.DataFrame(index=idx)
xrp_btc["price"] = xrp_usd["price"] / btc_usd["price"]
xrp_btc["return"] = xrp_btc["price"].pct_change()*100

btc_hist = compute_ind(btc_usd.copy())
xrp_hist = compute_ind(xrp_btc.copy())

# ---------------- Bootstrap ML ----------------
try:
    Xb, yb = gen_sig(btc_hist, True, btc_params)
    Xx, yx = gen_sig(xrp_hist, False, xrp_params)
    pieces, ys = [], []
    if len(Xb): pieces.append(Xb); ys.append(yb)
    if len(Xx): pieces.append(Xx); ys.append(yx)
    if pieces:
        X0 = np.vstack(pieces); y0 = np.concatenate(ys)
        st.session_state.online_clf.partial_fit(X0, y0)
except Exception as e:
    diag_msgs.append(f"ML bootstrap warning: {e}")

# ---------------- Live ----------------
live = load_live()
src = live.get("source","none")
if src != "none":
    diag_msgs.append(f"Live prices sourced from: {src}")
for e in live.get("errors",[]): diag_msgs.append(f"Live fetch note: {e}")

btc_p, btc_ch = live["BTC"]
xrp_p, _ = live["XRP"]
if np.isnan(btc_p):
    btc_p = btc_hist["price"].iat[-1]
    diag_msgs.append("Using last historical BTC close (live fetch failed).")
if np.isnan(xrp_p):
    xrp_p = xrp_hist["price"].iat[-1]
    diag_msgs.append("Using last historical XRP/BTC close (live fetch failed).")

st.sidebar.header("🧾 Logs / Diagnostics")
if diag_msgs:
    for m in diag_msgs: st.sidebar.write("- " + m)
else:
    st.sidebar.write("No diagnostic messages. History + live loaded OK.")

# ---------------- Scenario augmentation ----------------
for prices,is_b in [
    (list(btc_hist["price"].values[-90:]) + [btc_p], True),
    (list(xrp_hist["price"].values[-90:]) + [xrp_p], False)
]:
    Xr, yr = extract_Xy(prices, is_b)
    if Xr.size and yr.size:
        st.session_state.mem_X += Xr.tolist()
        st.session_state.mem_y += yr.tolist()
        st.session_state.mem_ts += [time.time()] * len(yr)

for is_b, vol in [(True, btc_hist["vol14"].iat[-1]), (False, xrp_hist["vol14"].iat[-1])]:
    for reg in ("normal","high-vol","crash","rally","flash-crash"):
        pr = generate_scenario(vol, reg)
        Xs, ys = extract_Xy(pr, is_b)
        if Xs.size and ys.size:
            st.session_state.mem_X += Xs.tolist()
            st.session_state.mem_y += ys.tolist()
            st.session_state.mem_ts += [0] * len(ys)

now = time.time()
keep = [i for i,t in enumerate(st.session_state.mem_ts) if t==0 or now - t <= 60*86400]
if len(keep) > 5000: keep = keep[-5000:]
st.session_state.mem_X = [st.session_state.mem_X[i] for i in keep]
st.session_state.mem_y = [st.session_state.mem_y[i] for i in keep]
st.session_state.mem_ts = [st.session_state.mem_ts[i] for i in keep]

buf_len = len(st.session_state.mem_y)
real_ct = sum(1 for t in st.session_state.mem_ts if t>0)
if buf_len>0 and real_ct / buf_len >= 0.10:
    try:
        bs = min(200, buf_len)
        idxs = random.sample(range(buf_len), bs)
        Xb = np.array([st.session_state.mem_X[i] for i in idxs])
        yb = np.array([st.session_state.mem_y[i] for i in idxs])
        if len(Xb):
            st.session_state.online_clf.partial_fit(Xb, yb)
    except Exception:
        pass

# ---------------- Predictions ----------------
def today_feat(df):
    if len(df) == 0: return [[0,0,0,0,0]]
    i = len(df)-1
    return [[df["rsi"].iat[i], df["vol14"].iat[i],
             df["price"].iat[i] - df["ema50"].iat[i],
             df["sma5"].iat[i] - df["sma20"].iat[i],
             df["return"].iat[i]]]

try: p_b = st.session_state.online_clf.predict_proba(today_feat(btc_hist))[:,1][0]
except Exception: p_b = 0.0
try: p_x = st.session_state.online_clf.predict_proba(today_feat(xrp_hist))[:,1][0]
except Exception: p_x = 0.0

# ---------------- Grid & bot helpers ----------------
def compute_grid_levels(low, up, n, spacing):
    if n <= 1: return [low, up]
    if spacing == "Arithmetic":
        step = (up - low) / n
        return [low + step * i for i in range(n+1)]
    else:
        if low <= 0:
            step = (up - low) / n
            return [low + step * i for i in range(n+1)]
        ratio = (up/low) ** (1.0 / n)
        return [low * (ratio ** i) for i in range(n+1)]

def regime_ok(df, prob):
    rsi_bound = BASE_RSI_OB + min(10, df["vol14"].iat[-1] * 100)
    return {
        "Price>EMA50": df["price"].iat[-1] > df["ema50"].iat[-1],
        "SMA5>SMA20": df["sma5"].iat[-1] > df["sma20"].iat[-1],
        "RSI<Bound": df["rsi"].iat[-1] < rsi_bound,
        "Vol≥Floor": df["vol14"].iat[-1] >= MIN_VOL,
        "ML Prob": prob >= CLASS_THRESH
    }

def compute_drop(df, pr, chg):
    vol = df["vol14"].iat[-1]
    ret = chg if chg is not None else df["return"].iat[-1]
    if pd.isna(vol) or pd.isna(ret) or vol <= 0 or ret < vol: return None
    return vol if ret <= 2*vol else 2*vol

def auto_state(key, hist, price, chg, prob, low_c, up_c, cnt_c):
    bal = st.session_state.bal_b if key == "b" else st.session_state.bal_x
    dep = st.session_state.get(f"deployed_{key}", False)
    term = st.session_state.get(f"terminated_{key}", False)
    raw_drop = hist["vol14"].iat[-1] if (st.session_state.mode == "new" and not dep) else compute_drop(hist, price, chg)
    drop = raw_drop if (raw_drop is not None and raw_drop > 0) else MIN_VOL

    if term and all(regime_ok(hist, prob).values()):
        st.session_state[f"terminated_{key}"] = False
        term = False
    if not dep and not term and all(regime_ok(hist, prob).values()):
        st.session_state[f"deployed_{key}"] = True
        dep = True

    if st.session_state.mode == "new":
        low = price * (1 - drop/100)
        up = price * (1 + max(0.001, drop*0.5/100))
    else:
        low = low_c; up = up_c

    if low is None or up is None or low<=0 or up<=0 or up<=low:
        low = low_c if low_c is not None else price * (1 - drop/100)
        up = up_c if up_c is not None else price * (1 + max(0.001, drop*0.5/100))

    sl = price * (1 - stop_loss/100)
    tp = up * (1 + (drop*1.5/100)) if drop else up
    rec = max(5, min(100, int((bal / max(price,1e-8)) // ((usd_tot/30)/ max(price,1e-8)))))
    grids = cnt_c if st.session_state.mode=="cont" else (manual_b if key=="b" and override else manual_x if key=="x" and override else rec)

    today = hist["price"].iat[-1]
    if not dep: act = "Not Deployed"
    elif term: act = "Terminated"
    elif today >= tp and tp > price: act = "Take-Profit"
    elif today <= sl: act = "Stop-Loss"
    elif dep and drop: act = "Redeploy"
    else: act = "Hold"

    if compound and act in ("Take-Profit","Stop-Loss"):
        factor = (1 + drop*1.5/100) if act=="Take-Profit" else (1 - stop_loss/100)
        if key == "b": st.session_state.bal_b *= factor
        else: st.session_state.bal_x *= factor

    return low, up, tp, sl, grids, rec, act

def format_price(v):
    try:
        if v is None: return "N/A"
        if isinstance(v, float) and np.isnan(v): return "N/A"
        if abs(v) < 1:
            return f"{v:.8f}"
        return f"{v:,.2f}"
    except Exception:
        return str(v)

# ---------------- Render bots (compact) ----------------
for key, label, hist, (pr,ch), prob in [
    ("b","🟡 BTC/USDT", btc_hist, (btc_p, btc_ch), p_b),
    ("x","🟣 XRP/BTC",   xrp_hist, (xrp_p, None),   p_x)
]:
    if st.session_state.mode == "cont":
        low_c = st.session_state.get(f"cont_low_{key}", pr)
        up_c  = st.session_state.get(f"cont_up_{key}", pr)
        cnt_c = st.session_state.get(f"cont_grids_{key}", 30)
    else:
        low_c = pr; up_c = pr; cnt_c = 30

    low, up, tp, sl, grids, rec, act = auto_state(key, hist, pr, ch, prob, low_c, up_c, cnt_c)

    c1, c2, c3 = st.columns([2,2,3])
    c1.markdown(f"### {label}")
    c2.markdown(f"**Price**  \n`{format_price(hist['price'].iat[-1])}`")
    if act == "Redeploy":
        c3.success(f"🔔 {act}")
    elif act == "Take-Profit":
        c3.info(f"💰 {act}")
    elif act == "Stop-Loss":
        c3.error(f"🔻 {act}")
    elif act == "Terminated":
        c3.error(f"🛑 {act}")
    elif act == "Not Deployed":
        c3.warning(act)
    else:
        c3.write(act)

    # Show compact sample always; full list only inside collapsed expander
    if act == "Redeploy":
        st.markdown("**Deploy Instructions — Grid Setup**")
        st.write(f"Lower (Low): `{format_price(low)}`  —  Upper (Up): `{format_price(up)}`  —  **Recommended grids:** {rec}")
        st.write(f"Spacing: **{st.session_state.spacing_type}**")

        try:
            n = int(grids) if grids and grids>0 else rec
            levels = compute_grid_levels(low, up, n, st.session_state.spacing_type)

            # Build a compact sample (first 3, last 3) regardless of total length.
            # If total < 7, show first and last and middle items in sample, still do NOT print full list outside expander.
            L = len(levels)
            if L <= 6:
                # show first, middle (if exists), last as sample (keeps display compact)
                sample = []
                for i in range(min(3, L)):
                    sample.append(format_price(levels[i]))
                if L > 3:
                    sample.append("...")
                    sample += [format_price(levels[-2]), format_price(levels[-1])]
                st.write(f"Grid sample ({L} levels):", sample)
            else:
                sample = [format_price(levels[0]), format_price(levels[1]), format_price(levels[2]), "...",
                          format_price(levels[-3]), format_price(levels[-2]), format_price(levels[-1])]
                st.write(f"Grid sample ({L} levels):", sample)

            # Full list is inside a collapsed expander (hidden by default)
            with st.expander("Show full grid levels (hidden)", expanded=False):
                st.write([format_price(x) for x in levels])

        except Exception as e:
            st.write("Could not compute grid levels:", e)

        st.caption("Only deploy when you confirm these bands in Crypto.com Grid UI.")
        if st.button(f"Mark {label} as Deployed"):
            st.session_state[f"deployed_{key}"] = True
            st.session_state[f"terminated_{key}"] = False
            st.success(f"{label} marked as Deployed (local)")

    else:
        st.write("")

    with st.expander("🔧 Diagnostics (advanced)", expanded=False):
        for k,v in regime_ok(hist, prob).items():
            st.write(f"{k}: {'✅' if v else '❌'}")
        st.write(f"ML Prob: {prob:.2f} ≥ {CLASS_THRESH}")
        try:
            snap = hist[["price","sma5","sma20","rsi","vol14"]].tail(3).copy()
            def fmt_series(s):
                if s.abs().max() <= 0.01:
                    return s.map(lambda x: f"{x:.8f}" if pd.notna(x) else "")
                return s.map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
            snap_display = pd.DataFrame({
                "Price": fmt_series(snap["price"]),
                "SMA5": fmt_series(snap["sma5"]),
                "SMA20": fmt_series(snap["sma20"]),
                "RSI": snap["rsi"].map(lambda x: f"{x:.4f}" if pd.notna(x) else ""),
                "Vol14": snap["vol14"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
            }, index=snap.index)
            snap_display.index = [d.strftime("%Y-%m-%d") for d in snap_display.index]
            st.table(snap_display)
        except Exception:
            pass

    flag = f"notified_{key}"
    st.session_state.setdefault(flag, False)
    if st.session_state.email_alerts and act in ("Redeploy","Take-Profit","Stop-Loss") and not st.session_state[flag]:
        try:
            subj = f"[Grid Bot] {label} Signal: {act}"
            body = f"{label} action: {act}\nPrice: {format_price(hist['price'].iat[-1])}\nLow: {format_price(low)}\nUp: {format_price(up)}\nTP: {format_price(tp)}\nSL: {format_price(sl)}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            msg = MIMEText(body)
            msg["Subject"] = subj
            msg["From"] = st.session_state.email_addr
            msg["To"] = st.session_state.email_addr
            s = smtplib.SMTP("smtp.mail.yahoo.com", 587, timeout=20)
            s.starttls()
            s.login(st.session_state.email_addr, st.session_state.email_pass)
            s.send_message(msg)
            s.quit()
            st.session_state[flag] = True
        except Exception as e:
            st.sidebar.write(f"Email send failed: {e}")

# ---------------- Persist ----------------
persist_all()
