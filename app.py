# app.py
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

# ─────────────── Persistence Helpers ───────────────
STATE_DIR   = ".state"
ML_BUF_FILE = os.path.join(STATE_DIR, "ml_buffer.pkl")
FLAGS_FILE  = os.path.join(STATE_DIR, "flags.pkl")

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

# ───────── Config / Constants ─────────
H_DAYS, VOL_W, RSI_W, EMA_T = 90, 14, 14, 50
BASE_RSI_OB, MIN_VOL        = 75, 0.5
CLASS_THRESH                = 0.80
MAX_RETRIES                 = 5

# ───────── Networking helpers (robust) ─────────
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
                time.sleep((backoff_base ** i) * 0.5)
                continue
        except requests.RequestException:
            time.sleep((backoff_base ** i) + random.random())
            continue
    return None

@st.cache_data(ttl=600)
def load_hist(coin, vs):
    """Attempt /range first (precise), fallback to /market_chart."""
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

@st.cache_data(ttl=60)
def load_live():
    def one(cid, vs, extra):
        j = fetch_json("https://api.coingecko.com/api/v3/simple/price",
                       {"ids": cid, "vs_currencies": vs, **(extra or {})})
        return j or {}
    try:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            b = ex.submit(one, "bitcoin", "usd", {"include_24hr_change": "true"})
            x = ex.submit(one, "ripple", "btc", {"include_24hr_change": "false"})
        j1, j2 = b.result(), x.result()
        btc = j1.get("bitcoin", {})
        xrp = j2.get("ripple", {})
        return {
            "BTC": (btc.get("usd", np.nan), btc.get("usd_24h_change", np.nan)),
            "XRP": (xrp.get("btc", np.nan), None)
        }
    except Exception:
        return {"BTC": (np.nan, None), "XRP": (np.nan, None)}

# ───────── Indicators & Feature Builders ─────────
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

# ───────── Streamlit Setup & State ─────────
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

# online classifier bootstrap placeholder (will be properly bootstrapped once history loads)
if "online_clf" not in st.session_state:
    clf = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
    clf.partial_fit(np.zeros((2, 5)), [0, 1], classes=[0, 1])
    st.session_state.online_clf = clf

# ───────── Sidebar controls (Persistence & Strategy) ─────────
st.sidebar.header("📣 Alerts & Notifications")
send_email_toggle = st.sidebar.checkbox("Enable Email Alerts", value=False)
email_addr = st.sidebar.text_input("Yahoo Email Address", value="")
email_pass = st.sidebar.text_input("Yahoo App Password", type="password")
st.session_state.email_alerts = send_email_toggle
st.session_state.email_addr = email_addr
st.session_state.email_pass = email_pass

st.sidebar.header("⚙️ Persistence & Resets")
if st.sidebar.button("🔄 Reset Bot State"):
    for b in ("b", "x"):
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
    for k in ("deployed_b", "terminated_b", "deployed_x", "terminated_x",
              "bal_b", "bal_x", "mem_X", "mem_y", "mem_ts", "mode"):
        st.session_state.pop(k, None)
    st.sidebar.success("All state deleted.")
    try:
        st.experimental_rerun()
    except Exception:
        st.stop()

st.sidebar.header("💰 Strategy Settings")
usd_tot = st.sidebar.number_input("Total Investment ($)", 100.0, 1e6, 3000.0, 100.0)
pct_btc = st.sidebar.slider("BTC Allocation (%)", 0, 100, 70)
usd_btc = usd_tot * pct_btc / 100
usd_xrp = usd_tot - usd_btc
gbp_rate = st.sidebar.number_input("GBP/USD Rate", 1.10, 1.60, 1.27, 0.01)
st.sidebar.metric("Portfolio", f"${usd_tot:,.2f}", f"£{usd_tot/gbp_rate:,.2f}")
stop_loss = st.sidebar.slider("Stop-Loss (%)", 1.0, 5.0, 2.0, 0.1)
compound = st.sidebar.checkbox("Enable Compounding", value=False)
mode = st.sidebar.radio("Mode", ["Start New Cycle", "Continue Existing"],
                        index=0 if st.session_state.get("mode", "new") == "new" else 1)
st.session_state.mode = "new" if mode == "Start New Cycle" else "cont"
override = st.sidebar.checkbox("Manual Grid Override", value=False)
manual_b = st.sidebar.number_input("BTC Grids", 2, 30, 6) if (override and st.session_state.mode == "new") else None
manual_x = st.sidebar.number_input("XRP Grids", 2, 30, 8) if (override and st.session_state.mode == "new") else None

if st.session_state.get("bal_b") is None:
    st.session_state.bal_b = usd_btc
if st.session_state.get("bal_x") is None:
    st.session_state.bal_x = usd_xrp

# ───────── Load History ─────────
btc_usd = load_hist("bitcoin", "usd")
xrp_usd = load_hist("ripple", "usd")
if btc_usd.empty or xrp_usd.empty:
    st.error("❌ Failed to load 90-day history; no 'price' data.")
    st.stop()

# align index and build xrp/btc price series
idx = btc_usd.index.intersection(xrp_usd.index)
btc_usd = btc_usd.reindex(idx)
xrp_usd = xrp_usd.reindex(idx)
xrp_btc = pd.DataFrame(index=idx)
xrp_btc["price"] = xrp_usd["price"] / btc_usd["price"]
xrp_btc["return"] = xrp_btc["price"].pct_change() * 100

# compute indicators
btc_hist = compute_ind(btc_usd.copy())
xrp_hist = compute_ind(xrp_btc.copy())

# ───────── Bootstrap classifier ONCE using real history (if not present) ─────────
if "online_clf" not in st.session_state or st.session_state.online_clf is None:
    clf = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
    Xb, yb = gen_sig(btc_hist, True, btc_params)
    Xx, yx = gen_sig(xrp_hist, False, xrp_params)
    pieces = []
    ys_pieces = []
    if len(Xb):
        pieces.append(Xb); ys_pieces.append(yb)
    if len(Xx):
        pieces.append(Xx); ys_pieces.append(yx)
    if pieces:
        X0 = np.vstack(pieces)
        y0 = np.concatenate(ys_pieces)
    else:
        X0 = np.zeros((2, 5))
        y0 = np.array([0, 1])
    clf.partial_fit(X0, y0, classes=[0, 1])
    st.session_state.online_clf = clf

# ───────── Live price fetch with fallback ─────────
live = load_live()
btc_p, btc_ch = live["BTC"]
xrp_p, _ = live["XRP"]
if np.isnan(btc_p) or np.isnan(xrp_p):
    st.warning("⚠️ Could not fetch live prices; using last historical close in computations.")
    btc_p = btc_hist["price"].iat[-1]
    xrp_p = xrp_hist["price"].iat[-1]
    btc_ch = None

# ───────── Scenario Augmentation (real today + synthetic) ─────────
# append real today examples
for prices, is_b in [
    (list(btc_hist["price"].values[-90:]) + [btc_p], True),
    (list(xrp_hist["price"].values[-90:]) + [xrp_p], False)
]:
    Xr, yr = extract_Xy(prices, is_b)
    if Xr.size and yr.size:
        st.session_state.mem_X += Xr.tolist()
        st.session_state.mem_y += yr.tolist()
        st.session_state.mem_ts += [time.time()] * len(yr)

# append synthetic scenarios safely
for is_b, vol in [(True, btc_hist["vol14"].iat[-1]), (False, xrp_hist["vol14"].iat[-1])]:
    for reg in ("normal", "high-vol", "crash", "rally", "flash-crash"):
        pr = generate_scenario(vol, reg, days=90)
        Xs, ys = extract_Xy(pr, is_b)
        if Xs.size and ys.size:
            st.session_state.mem_X += Xs.tolist()
            st.session_state.mem_y += ys.tolist()
            st.session_state.mem_ts += [0] * len(ys)

# trim buffer & expiry (60 days)
now = time.time()
keep = [i for i, t in enumerate(st.session_state.mem_ts) if t == 0 or now - t <= 60 * 86400]
if len(keep) > 5000:
    keep = keep[-5000:]
st.session_state.mem_X = [st.session_state.mem_X[i] for i in keep]
st.session_state.mem_y = [st.session_state.mem_y[i] for i in keep]
st.session_state.mem_ts = [st.session_state.mem_ts[i] for i in keep]

# online partial_fit if real fraction ≥ 10%
buf_len = len(st.session_state.mem_y)
real_ct = sum(1 for t in st.session_state.mem_ts if t > 0)
if buf_len > 0 and real_ct / buf_len >= 0.10:
    bs = min(200, buf_len)
    idxs = random.sample(range(buf_len), bs)
    Xb = np.array([st.session_state.mem_X[i] for i in idxs])
    yb = np.array([st.session_state.mem_y[i] for i in idxs])
    if len(Xb):
        try:
            st.session_state.online_clf.partial_fit(Xb, yb)
        except Exception:
            # ignore partial_fit failures (shapes, etc.) to keep UI alive
            pass

# ───────── Predictions ─────────
try:
    p_b = st.session_state.online_clf.predict_proba(today_feat(btc_hist))[:, 1][0]
except Exception:
    p_b = 0.0
try:
    p_x = st.session_state.online_clf.predict_proba(today_feat(xrp_hist))[:, 1][0]
except Exception:
    p_x = 0.0

# ───────── Bot Logic & Render ─────────
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
    if pd.isna(vol) or pd.isna(ret) or vol <= 0 or ret < vol:
        return None
    return vol if ret <= 2 * vol else 2 * vol

def auto_state(key, hist, price, chg, prob, low_c, up_c, cnt_c):
    bal = st.session_state.bal_b if key == "b" else st.session_state.bal_x
    dep = st.session_state.get(f"deployed_{key}", False)
    term = st.session_state.get(f"terminated_{key}", False)
    drop = hist["vol14"].iat[-1] if (st.session_state.mode == "new" and not dep) else compute_drop(hist, price, chg)

    # recover
    if term and all(regime_ok(hist, prob).values()):
        st.session_state[f"terminated_{key}"] = False
        term = False

    # deploy
    if not dep and not term and all(regime_ok(hist, prob).values()):
        st.session_state[f"deployed_{key}"] = True
        dep = True

    low = price * (1 - drop / 100) if (st.session_state.mode == "new" and drop) else low_c
    up = price if st.session_state.mode == "new" else up_c
    sl = price * (1 - stop_loss / 100)
    tp = up * (1 + (drop * 1.5 / 100)) if drop else up_c

    rec = max(5, min(30, int((bal / max(price, 1e-8)) // ((usd_tot / 30) / max(price, 1e-8)))))
    grids = cnt_c if st.session_state.mode == "cont" else (
        manual_b if key == "b" and override else
        manual_x if key == "x" and override else
        rec
    )

    today = hist["price"].iat[-1]
    if not dep:
        act = "Not Deployed"
    elif term:
        act = "Terminated"
    elif today >= tp:
        act = "Take-Profit"
    elif today <= sl:
        act = "Stop-Loss"
    elif dep and drop:
        act = "Redeploy"
    else:
        act = "Hold"

    if compound and act in ("Take-Profit", "Stop-Loss"):
        factor = (1 + drop * 1.5 / 100) if act == "Take-Profit" else (1 - stop_loss / 100)
        if key == "b":
            st.session_state.bal_b *= factor
        else:
            st.session_state.bal_x *= factor

    return low, up, tp, sl, grids, rec, act

# Render each bot
for key, label, hist, (pr, ch), prob in [
    ("b", "🟡 BTC/USDT", btc_hist, (btc_p, btc_ch), p_b),
    ("x", "🟣 XRP/BTC", xrp_hist, (xrp_p, None), p_x)
]:
    low, up, tp, sl, grids, rec, act = auto_state(
        key, hist, pr, ch, prob,
        st.session_state.get(f"cont_low_{key}", pr),
        st.session_state.get(f"cont_up_{key}", pr),
        st.session_state.get(f"cont_grids_{key}", 30)
    )
    st.subheader(f"{label} Bot")
    diag = regime_ok(hist, prob)
    with st.expander("🔧 Diagnostics", expanded=False):
        for k, v in diag.items():
            st.write(f"{k}: {'✅' if v else '❌'}")
        st.write(f"ML Prob: {prob:.2f} ≥ {CLASS_THRESH}")

        # Extra numeric diagnostics (SMA values & historical counts)
        try:
            sma5_now = hist["sma5"].iat[-1]
            sma20_now = hist["sma20"].iat[-1]
            st.write(f"Current SMA5: **{sma5_now:.6f}** · Current SMA20: **{sma20_now:.6f}**")
        except Exception:
            st.write("SMA values not available")

        try:
            valid = hist.dropna(subset=["sma5", "sma20"])
            if len(valid):
                n_cross = (valid["sma5"] > valid["sma20"]).sum()
                pct = 100 * n_cross / len(valid)
                st.write(f"SMA5>SMA20 historically: **{n_cross}** of **{len(valid)}** days ({pct:.1f}%)")
                st.table(valid[["price", "sma5", "sma20"]].tail(6).rename(columns={"price":"Price","sma5":"SMA5","sma20":"SMA20"}))
            else:
                st.write("Not enough SMA history to show stats.")
        except Exception:
            pass

    # Action panel
    if act == "Redeploy":
        st.info("🔔 Redeploy signal")
    elif act == "Take-Profit":
        st.success("💰 TAKE-PROFIT")
    elif act == "Stop-Loss":
        st.error("🔻 STOP-LOSS")
    elif act == "Terminated":
        st.error("🛑 TERMINATED")
    else:
        st.info("⏸ HOLD")

    # Email notification (one-shot per event)
    notified_flag = f"notified_{key}"
    st.session_state.setdefault(notified_flag, False)
    if st.session_state.email_alerts and act in ("Redeploy", "Take-Profit", "Stop-Loss") and not st.session_state[notified_flag]:
        try:
            # Basic Yahoo SMTP send (user must supply app password)
            subject = f"[Grid Bot] {label} Signal: {act}"
            body = f"{label} bot action: {act} at price {hist['price'].iat[-1]:.6f} on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = st.session_state.email_addr
            msg["To"] = st.session_state.email_addr
            server = smtplib.SMTP("smtp.mail.yahoo.com", 587, timeout=20)
            server.starttls()
            server.login(st.session_state.email_addr, st.session_state.email_pass)
            server.send_message(msg)
            server.quit()
            st.session_state[notified_flag] = True
        except Exception as e:
            st.warning(f"Email send failed: {e}")

# ───────── Persist & Exit ─────────
persist_all()
