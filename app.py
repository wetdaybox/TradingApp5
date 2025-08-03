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

# ── Constants ──
HISTORY_DAYS      = 90
VOL_WINDOW        = 14
RSI_WINDOW        = 14
EMA_TREND         = 50
GRID_PRIMARY      = 20
GRID_MAX          = 30
CLASS_PROB_THRESH = 0.80
MAX_RETRIES       = 3

# ── Network Helpers ──
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
def load_all_live():
    def fetch_one(key, coin_id, vs, extra):
        data = fetch_json(
            "https://api.coingecko.com/api/v3/simple/price",
            {"ids":coin_id, "vs_currencies":vs, **extra}
        ) or {}
        return key, data

    with concurrent.futures.ThreadPoolExecutor() as exe:
        futures = {
            exe.submit(fetch_one, "BTC", "bitcoin", "usd", {"include_24hr_change":"true"}): "BTC",
            exe.submit(fetch_one, "XRP", "ripple", "btc", {}): "XRP"
        }
        out = {}
        for fut in concurrent.futures.as_completed(futures):
            key = futures[fut]
            data = fut.result()[1]
            if key=="BTC":
                b = data.get("bitcoin",{})
                out["BTC"] = (b.get("usd", np.nan), b.get("usd_24h_change", np.nan))
            else:
                x = data.get("ripple",{})
                out["XRP"] = (x.get("btc", np.nan), None)
        return out

# ── Signal & ML Helpers ──
def gen_signals(df, is_btc, params):
    X,y = [],[]
    for i in range(EMA_TREND, len(df)-1):
        p,ret,vol = df["price"].iat[i], df["return"].iat[i], df["vol14"].iat[i]
        ema_diff   = p - df["ema50"].iat[i]
        mom        = df["sma5"].iat[i] - df["sma20"].iat[i]
        rsi        = df["rsi"].iat[i]
        if is_btc:
            rsi_th,_,_ = params
            cond = (ema_diff>0) and (mom>0) and (rsi<rsi_th) and (ret>=vol)
        else:
            m,b,sl,dip = params
            mval = df["price"].rolling(m).mean().iat[i]
            cond = (p<mval) and (((mval-p)/p*100)>=dip) and (vol>df["vol14"].iat[i-1])
        if not cond:
            continue
        X.append([rsi,vol,ema_diff,mom,ret])
        profit = df["price"].iat[i+1] - p
        y.append(1 if profit>0 else 0)
    return np.array(X), np.array(y)

@st.cache_resource
def get_trained_models(Xb,yb,Xx,yx):
    def train_once(X,y):
        if len(y)>=6 and len(np.unique(y))>1:
            gs = GridSearchCV(
                RandomForestClassifier(random_state=0),
                {"n_estimators":[50,100], "max_depth":[3,5]},
                cv=3, scoring="accuracy", n_jobs=1
            )
            gs.fit(X,y)
            return gs.best_estimator_
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        if len(y)>0:
            clf.fit(X,y)
        return clf
    return train_once(Xb,yb), train_once(Xx,yx)

def today_feat(df):
    if df.empty:
        return None
    i = len(df)-1
    return [[
        df["rsi"].iat[i],
        df["vol14"].iat[i] if "vol14" in df.columns else np.nan,
        df["price"].iat[i]-df["ema50"].iat[i],
        df["sma5"].iat[i]-df["sma20"].iat[i],
        df["return"].iat[i],
    ]]

def safe_prob(clf, feat):
    if feat is None:
        return 0.0
    probs = clf.predict_proba(feat)[0]
    return probs[1] if probs.shape[0]>1 else 0.0

# ── Initialization ──
with st.spinner("🚀 Initializing data & models…"):
    btc_hist      = load_history("bitcoin","usd")
    xrp_hist      = load_history("ripple","btc")
    live          = load_all_live()
    (btc_p,btc_ch)= live["BTC"]
    (xrp_p,_)     = live["XRP"]

    btc_params = (75,1.5,1.0)
    xrp_params = (10,75,50,1.0)

    Xb,yb = gen_signals(btc_hist, True,  btc_params)
    Xx,yx = gen_signals(xrp_hist, False, xrp_params)

    clf_btc, clf_xrp = get_trained_models(Xb,yb,Xx,yx)

    p_btc   = safe_prob(clf_btc, today_feat(btc_hist))
    p_xrp   = safe_prob(clf_xrp, today_feat(xrp_hist))
    use_btc = p_btc >= CLASS_PROB_THRESH
    use_xrp = p_xrp >= CLASS_PROB_THRESH

# ── Sidebar ──
st.sidebar.header("💰 Investment")
usd_total     = st.sidebar.number_input("Total Investment ($)",100.0,1e6,3000.0,100.0)
pct_btc       = st.sidebar.slider("BTC Allocation (%)",0,100,70)
usd_btc_alloc = usd_total * pct_btc/100
usd_xrp_alloc = usd_total - usd_btc_alloc
gbp_rate      = st.sidebar.number_input("GBP/USD Rate",1.10,1.60,1.27,0.01)
st.sidebar.metric("Total Value (USD/GBP)", f"${usd_total:,.2f}", f"£{usd_total/gbp_rate:,.2f}")
min_order     = st.sidebar.number_input("Min Order (BTC)",1e-6,1e-2,5e-4,1e-6,format="%.6f")
MIN_ORDER     = max(min_order, (usd_btc_alloc/GRID_MAX)/btc_p if btc_p else 0)
st.sidebar.caption(f"Min Order ≥ {MIN_ORDER:.6f} BTC (~${MIN_ORDER*btc_p:.2f})")

# ── Compute Drops & Grid Levels ──
def compute_drop(df, price, change):
    if df.empty:
        return 0
    vol14 = df["vol14"].iat[-1] if "vol14" in df.columns else np.nan
    ret24 = change if change is not None else df["return"].iat[-1]
    if np.isnan(vol14) or ret24 < vol14:
        return 0
    return vol14 if ret24 <= 2*vol14 else 2*vol14

drop_btc = compute_drop(btc_hist, btc_p, btc_ch)
drop_xrp = xrp_params[1] if (
    not xrp_hist.empty
    and xrp_hist["price"].iat[-1] < xrp_hist["price"].rolling(int(xrp_params[0])).mean().iat[-1]
    and xrp_hist["vol14"].iat[-1] > xrp_hist["vol14"].iat[-2]
) else 0

levels_b = GRID_PRIMARY if not use_btc else GRID_MAX
levels_x = GRID_PRIMARY if not use_xrp else GRID_MAX

lower_b = btc_p * (1 - drop_btc/100)
upper_b = btc_p
tp_b    = upper_b * (1 + drop_btc/100)
action_b= "Redeploy" if drop_btc>0 else "Terminate"

lower_x = xrp_p * (1 - drop_xrp/100)
upper_x = xrp_p
tp_x    = upper_x * (1 + drop_xrp/100)
action_x= "Redeploy" if drop_xrp>0 else "Terminate"

# ── Display Function ──
def show_grid_bot(title, grids, lower, upper, tp, action, key):
    st.subheader(title)

    # If no valid grid or termination, show friendly message
    if np.isnan(lower) or action == "Terminate":
        st.info("⚠️ No grid reset recommended at this time.")
        st.write(f"**Action:** { 'Terminate / Hold' if action=='Terminate' else action }")
        return

    # Otherwise show grid parameters
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Grids",       f"{grids}")
        st.metric("Lower Price", f"{lower:,.6f}")
        st.metric("Upper Price", f"{upper:,.6f}")
        st.metric("Take-Profit", f"{tp:,.6f}")
    with c2:
        if st.button("🔄 Redeploy Now", key=f"{key}_redeploy"):
            st.success("✅ Copy Grids, Lower & Upper into Crypto.com Grid Box")

    # Details expander
    with st.expander("Details"):
        hist = btc_hist if "BTC" in title else xrp_hist
        prob = p_btc    if "BTC" in title else p_xrp
        lvl  = levels_b if "BTC" in title else levels_x

        vol14_val = (
            hist["vol14"].iat[-1]
            if "vol14" in hist.columns and len(hist) >= VOL_WINDOW
            else None
        )
        rsi_val = (
            hist["rsi"].iat[-1]
            if "rsi" in hist.columns and len(hist) >= RSI_WINDOW
            else None
        )

        st.write(f"- **Action:** {action}")
        if vol14_val is not None:
            st.write(f"- **14 d Volatility:** {vol14_val:.2f}%")
        if rsi_val is not None:
            st.write(f"- **14 d RSI:** {rsi_val:.1f}")
        st.write(f"- **ML Confidence:** {int(prob*100)}%")
        st.write(f"- **Grid Levels used:** {lvl}")

# ── Render Bots ──
show_grid_bot("🟡 BTC/USDT Bot", levels_b, lower_b, upper_b, tp_b, action_b, "btc")
show_grid_bot("🟣 XRP/BTC Bot", levels_x, lower_x, upper_x, tp_x, action_x, "xrp")

# ── About & Requirements ──
with st.expander("ℹ️ About & Usage"):
    st.write("""
    1. Copy **Grids**, **Lower Price**, **Upper Price** into Crypto.com Grid Box.  
    2. Click **Redeploy Now** when signaled, or **Terminate Bot** otherwise.  
    3. Click **Details** for volatility, RSI, ML confidence & grid count.  
    4. App auto-refreshes every 60 s.
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
