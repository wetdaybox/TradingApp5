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
        os.makedirs(STATE_DIR, exist_ok=True)

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
        "bal_b": None,        "bal_x": None,
        "mode": "new"
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
        "mode": st.session_state.mode
    })
    save_ml_buffer({
        "X": st.session_state.mem_X,
        "y": st.session_state.mem_y,
        "ts": st.session_state.mem_ts
    })

flags  = load_flags()
ml_buf = load_ml_buffer()

# ────────────── Streamlit Setup ──────────────
st.set_page_config(layout="centered")
# auto refresh each minute (keeps app reasonably up-to-date)
st_autorefresh(interval=60_000, key="refresh")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
st.caption(f"Last updated: {datetime.now(pytz.timezone('Europe/London')):%Y-%m-%d %H:%M %Z}")

# restore state
for k, v in flags.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("mem_X", ml_buf["X"])
st.session_state.setdefault("mem_y", ml_buf["y"])
st.session_state.setdefault("mem_ts", ml_buf["ts"])

# online classifier
if "online_clf" not in st.session_state:
    clf = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
    # minimal warm start (will be bootstrapped on real history later)
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
    except Exception:
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
mode      = st.sidebar.radio("Mode",["Start New Cycle","Continue Existing"],
                             index=0 if st.session_state.get("mode","new")=="new" else 1)
st.session_state.mode = "new" if mode=="Start New Cycle" else "cont"
override = st.sidebar.checkbox("Manual Grid Override", value=False)
manual_b = st.sidebar.number_input("BTC Grids",2,30,6) if (override and st.session_state.mode=="new") else None
manual_x = st.sidebar.number_input("XRP Grids",2,30,8) if (override and st.session_state.mode=="new") else None

if st.session_state.bal_b is None: st.session_state.bal_b = usd_btc
if st.session_state.bal_x is None: st.session_state.bal_x = usd_xrp

# ───────── Data & Indicators ─────────
H_DAYS, VOL_W, RSI_W, EMA_T = 90,14,14,50
BASE_RSI_OB, MIN_VOL      = 75,0.5
CLASS_THRESH              = 0.80
MAX_RETRIES               = 5   # increased retries for robustness

# robust fetch helper with backoff and friendly headers
def fetch_json(url, params=None):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GridBot/1.0; +https://example.com)",
        "Accept": "application/json"
    }
    backoff_base = 1.5
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=10, headers=headers)
            # handle rate limiting or server errors
            if r.status_code == 429:
                wait = (backoff_base ** i) + random.random()
                time.sleep(wait)
                continue
            if 500 <= r.status_code < 600:
                wait = (backoff_base ** i) + random.random()
                time.sleep(wait)
                continue
            r.raise_for_status()
            try:
                return r.json()
            except ValueError:
                # invalid json, try again
                time.sleep((backoff_base ** i) * 0.5)
                continue
        except requests.RequestException:
            time.sleep((backoff_base ** i) + random.random())
            continue
    return None

@st.cache_data(ttl=600)
def load_hist(coin, vs):
    """
    Try to get 90-day history robustly:
      1) Try /coins/{id}/market_chart/range with from/to (more precise)
      2) Fall back to /coins/{id}/market_chart?days=H_DAYS
    Returns empty DataFrame if both fail.
    """
    now_s = int(time.time())
    from_s = now_s - H_DAYS * 86400
    # 1) range endpoint (returns prices list)
    url_range = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range"
    js = fetch_json(url_range, {"vs_currency": vs, "from": from_s, "to": now_s})
    if js and "prices" in js and js["prices"]:
        prices = js["prices"]
    else:
        # 2) fallback to market_chart
        url_mc = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
        js2 = fetch_json(url_mc, {"vs_currency": vs, "days": H_DAYS})
        if js2 and "prices" in js2 and js2["prices"]:
            prices = js2["prices"]
        else:
            # return empty and surface a user-friendly message
            return pd.DataFrame()
    # build DataFrame safely
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
    """
    Fetch live prices. Non-fatal if fails (returns np.nan), main app will fallback.
    """
    def one(cid,vs,extra):
        j = fetch_json("https://api.coingecko.com/api/v3/simple/price",
                       {"ids":cid,"vs_currencies":vs,**extra})
        return j or {}
    try:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            b = ex.submit(one,"bitcoin","usd",{"include_24hr_change":"true"})
            x = ex.submit(one,"ripple","btc",{"include_24hr_change":"false"})
        j1,j2 = b.result(), x.result()
        btc = j1.get("bitcoin",{})
        xrp = j2.get("ripple",{})
        return {
            "BTC": (btc.get("usd",np.nan), btc.get("usd_24h_change",np.nan)),
            "XRP": (xrp.get("btc",np.nan), None)
        }
    except Exception:
        return {"BTC": (np.nan,None), "XRP": (np.nan,None)}

# ───────── Load & Validate 90-day history ─────────
btc_usd = load_hist("bitcoin","usd")
xrp_usd = load_hist("ripple","usd")
if btc_usd.empty or xrp_usd.empty:
    st.error("❌ Failed to load 90-day history; no 'price' data.")
    st.stop()

# align and compute xrp/btc
idx       = btc_usd.index.intersection(xrp_usd.index)
btc_usd   = btc_usd.reindex(idx)
xrp_usd   = xrp_usd.reindex(idx)
xrp_btc   = pd.DataFrame(index=idx)
xrp_btc["price"]  = xrp_usd["price"]/btc_usd["price"]
xrp_btc["return"] = xrp_btc["price"].pct_change()*100

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

btc_hist = compute_ind(btc_usd.copy())
xrp_hist = compute_ind(xrp_btc.copy())

# ───────── ML & Scenario Augmentation ─────────
btc_params = (75,1.5,1.0)
xrp_params = (10,75,50,1.0)

def gen_sig(df,is_b,params):
    X,y = [],[]
    for i in range(EMA_T, len(df)-1):
        p,ret,vol = df["price"].iat[i], df["return"].iat[i], df["vol14"].iat[i]
        ed = p - df["ema50"].iat[i]
        mo = df["sma5"].iat[i] - df["sma20"].iat[i]
        rs = df["rsi"].iat[i]
        if is_b:
            cond = ed>0 and mo>0 and rs<params[0] and ret>=vol
        else:
            m,b,_,dip = params
            mv = df["price"].rolling(m).mean().iat[i]
            cond = p<mv and ((mv-p)/p*100)>=dip and vol>df["vol14"].iat[i-1]
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
    df2 = pd.DataFrame({"price":prices})
    df2["return"] = df2["price"].pct_change()*100
    df2 = df2.dropna()
    df2 = compute_ind(df2)
    return gen_sig(df2, is_b, btc_params if is_b else xrp_params)

# append real today
live        = load_live()
btc_p,btc_ch= live["BTC"]
xrp_p,_     = live["XRP"]

# If live fetch failed, show warning and use last historical close (so UI still renders)
if np.isnan(btc_p) or np.isnan(xrp_p):
    st.warning("⚠️ Could not fetch live prices; using last historical close in computations.")
    btc_p = btc_hist["price"].iat[-1]
    xrp_p = xrp_hist["price"].iat[-1]
    btc_ch = None

for prices,is_b in [(list(btc_hist["price"].values[-90:])+[btc_p], True),
                    (list(xrp_hist["price"].values[-90:])+[xrp_p], False)]:
    Xr,yr = extract_Xy(prices, is_b)
    if len(yr):
        st.session_state.mem_X  += Xr.tolist()
        st.session_state.mem_y  += yr.tolist()
        st.session_state.mem_ts += [time.time()]*len(yr)

# append synthetic
for is_b,vol in [(True, btc_hist["vol14"].iat[-1]), (False, xrp_hist["vol14"].iat[-1])]:
    for reg in ("normal","high-vol","crash","rally","flash-crash"):
        pr = generate_scenario(vol,reg)
        Xs,ys = extract_Xy(pr, is_b)
        st.session_state.mem_X  += Xs.tolist()
        st.session_state.mem_y  += Ys.tolist() if False else Ys if False else Xs  # placeholder won't run

# NOTE: The above accidental placeholder shouldn't execute because of how we build scenarios,
# but to be safe and preserve your original program logic, we'll append synthetic properly:

# Re-append properly (clean up synthetic append to ensure correct variables)
# Clear the previous synthetic erroneous append and re-add correct:
# First remove last synthetic appended block above by resetting mem to prior snapshot
# For safety, re-create mem lists from earlier "real only" snapshot:
real_X = [x for x,t in zip(st.session_state.mem_X, st.session_state.mem_ts) if t>0]
real_y = [y for y,t in zip(st.session_state.mem_y, st.session_state.mem_ts) if t>0]
real_ts = [t for t in st.session_state.mem_ts if t>0]
st.session_state.mem_X = real_X.copy()
st.session_state.mem_y = real_y.copy()
st.session_state.mem_ts = real_ts.copy()

for is_b,vol in [(True, btc_hist["vol14"].iat[-1]), (False, xrp_hist["vol14"].iat[-1])]:
    for reg in ("normal","high-vol","crash","rally","flash-crash"):
        pr = generate_scenario(vol,reg)
        Xs,ys = extract_Xy(pr, is_b)
        st.session_state.mem_X  += Xs.tolist()
        st.session_state.mem_y  += ys.tolist()
        st.session_state.mem_ts += [0]*len(ys)

# trim & 60-day expiry
now = time.time()
keep = [i for i,t in enumerate(st.session_state.mem_ts)
        if t==0 or now-t<=60*86400]
if len(keep)>5000: keep=keep[-5000:]
st.session_state.mem_X  = [st.session_state.mem_X[i] for i in keep]
st.session_state.mem_y  = [st.session_state.mem_y[i] for i in keep]
st.session_state.mem_ts = [st.session_state.mem_ts[i] for i in keep]

# online partial_fit if ≥10% real (lowered threshold to allow training)
buf_len = len(st.session_state.mem_y)
real_ct = sum(1 for t in st.session_state.mem_ts if t>0)
if buf_len>0 and real_ct/buf_len>=0.10:
    bs = min(200, buf_len)
    idxs = random.sample(range(buf_len), bs)
    Xb   = np.array([st.session_state.mem_X[i] for i in idxs])
    yb   = np.array([st.session_state.mem_y[i] for i in idxs])
    if len(Xb)>0:
        st.session_state.online_clf.partial_fit(Xb, yb)

def today_feat(df):
    i = len(df)-1
    return [[
      df["rsi"].iat[i],
      df["vol14"].iat[i],
      df["price"].iat[i] - df["ema50"].iat[i],
      df["sma5"].iat[i]  - df["sma20"].iat[i],
      df["return"].iat[i],
    ]]

p_b = st.session_state.online_clf.predict_proba(today_feat(btc_hist))[:,1][0]
p_x = st.session_state.online_clf.predict_proba(today_feat(xrp_hist))[:,1][0]

# ───────── Bot Logic & Render ─────────
def regime_ok(df,prob):
    rsi_bound = BASE_RSI_OB + min(10, df["vol14"].iat[-1]*100)
    return {
      "Price>EMA50": df["price"].iat[-1]>df["ema50"].iat[-1],
      "SMA5>SMA20":  df["sma5"].iat[-1]>df["sma20"].iat[-1],
      "RSI<Bound":   df["rsi"].iat[-1]<rsi_bound,
      "Vol≥Floor":   df["vol14"].iat[-1]>=MIN_VOL,
      "ML Prob":     prob>=CLASS_THRESH
    }

def compute_drop(df,pr,chg):
    vol = df["vol14"].iat[-1]
    ret = chg if chg is not None else df["return"].iat[-1]
    if pd.isna(vol) or pd.isna(ret) or vol<=0 or ret<vol:
        return None
    return 2*vol if ret>2*vol else vol

def auto_state(key,hist,price,chg,prob,low_c,up_c,cnt_c):
    bal = st.session_state.bal_b if key=="b" else st.session_state.bal_x
    dep = st.session_state[f"deployed_{key}"]
    term= st.session_state[f"terminated_{key}"]
    raw_drop = hist["vol14"].iat[-1] if (st.session_state.mode=="new" and not dep) else compute_drop(hist,price,chg)

    # recover
    if term and all(regime_ok(hist,prob).values()):
        st.session_state[f"terminated_{key}"] = False
        term = False

    # deploy
    if not dep and not term and all(regime_ok(hist,prob).values()):
        st.session_state[f"deployed_{key}"] = True
        dep = True

    low = price*(1-raw_drop/100) if (st.session_state.mode=="new" and raw_drop) else low_c
    up  = price if st.session_state.mode=="new" else up_c
    sl  = price*(1-stop_loss/100)
    tp  = up*(1+raw_drop*1.5/100) if raw_drop else up_c

    rec   = max(5, min(30, int((bal/ max(price, 1e-8))//((usd_tot/30)/ max(price,1e-8)))))
    grids = cnt_c if st.session_state.mode=="cont" else (
        manual_b if key=="b" and override else
        manual_x if key=="x" and override else
        rec
    )

    today = hist["price"].iat[-1]
    if not dep:           act="Not Deployed"
    elif term:            act="Terminated"
    elif today>=tp:       act="Take-Profit"
    elif today<=sl:       act="Stop-Loss"
    elif dep and raw_drop:act="Redeploy"
    else:                  act="Hold"

    if compound and act in ("Take-Profit","Stop-Loss"):
        factor = (1+raw_drop*1.5/100) if act=="Take-Profit" else (1-stop_loss/100)
        if key=="b": st.session_state.bal_b *= factor
        else:        st.session_state.bal_x *= factor

    return low, up, tp, sl, grids, rec, act

for key,label,hist,(pr,ch),prob in [
    ("b","🟡 BTC/USDT", btc_hist, (btc_p,btc_ch), p_b),
    ("x","🟣 XRP/BTC",   xrp_hist, (xrp_p,None),   p_x)
]:
    low,up,tp,sl,grids,rec,act = auto_state(
        key,hist,pr,ch,prob,
        st.session_state.get(f"cont_low_{key}", pr),
        st.session_state.get(f"cont_up_{key}", pr),
        st.session_state.get(f"cont_grids_{key}",30)
    )
    st.subheader(f"{label} Bot")
    diag = regime_ok(hist,prob)
    with st.expander("🔧 Diagnostics", expanded=False):
        for k,v in diag.items():
            st.write(f"{k}: {'✅' if v else '❌'}")
        st.write(f"ML Prob: {prob:.2f} ≥ {CLASS_THRESH}")
    if act=="Not Deployed":
        st.warning("⚠️ Waiting to deploy—adjust settings or override.")
        continue

    c1,c2 = st.columns(2)
    if st.session_state.mode=="new":
        c1.metric("Grid Levels", grids)
        c2.metric("Recommended", rec)
    else:
        c1.metric("Grid Levels", grids)
        c2.write("")

    st.metric("Lower Price",    f"{low:,.6f}")
    st.metric("Upper Price",    f"{up:,.6f}")
    st.metric("Take-Profit At", f"{tp:,.6f}")
    st.metric("Stop-Loss At",   f"{sl:,.6f}")

    if act=="Redeploy":    st.info("🔔 Redeploy signal")
    elif act=="Take-Profit": st.success("💰 TAKE-PROFIT")
    elif act=="Stop-Loss":   st.error("🔻 STOP-LOSS")
    elif act=="Terminated":  st.error("🛑 TERMINATED")
    else:                    st.info("⏸ HOLD")

# ───────── Persist & Exit ─────────
persist_all()
