# app.py
import streamlit as st
import requests, time, concurrent.futures
import pandas as pd, numpy as np
from datetime import datetime
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from streamlit_autorefresh import st_autorefresh

# ── Auto-refresh & Page Setup ──
st_autorefresh(interval=60_000, key="refresh")
st.set_page_config(layout="centered")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
st.caption(f"Last updated: {datetime.now(pytz.timezone('Europe/London')):%Y-%m-%d %H:%M %Z}")

# ── Session-State Flags ──
for b in ("b", "x"):
    st.session_state.setdefault(f"deployed_{b}", False)
    st.session_state.setdefault(f"terminated_{b}", False)
st.session_state.setdefault("mode", None)

# ── Constants ──
H_DAYS, VOL_W, RSI_W, EMA_T = 90, 14, 14, 50
RSI_OB       = 75
MIN_VOL      = 0.5       # min daily vol% to consider trading
GRID_MAX     = 30
CLASS_THRESH = 0.80      # ML probability threshold
MAX_RETRIES  = 3

# ── HTTP Fetch Helpers ──
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
def load_hist_usd(coin):
    js = fetch_json(
        f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
        {"vs_currency":"usd","days":H_DAYS}
    ) or {}
    df = pd.DataFrame(js.get("prices", []), columns=["ts","price"])
    if df.empty: return df
    df["date"]   = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["price"]  = df["price"].astype(float)
    df["return"] = df["price"].pct_change() * 100
    return df

@st.cache_data(ttl=60)
def load_live():
    def one(cid, vs, extra):
        return fetch_json(
            "https://api.coingecko.com/api/v3/simple/price",
            {"ids":cid, "vs_currencies":vs, **extra}
        ) or {}
    with concurrent.futures.ThreadPoolExecutor() as ex:
        f1 = ex.submit(one, "bitcoin", "usd", {"include_24hr_change":"true"})
        f2 = ex.submit(one, "ripple", "btc", {"include_24hr_change":"false"})
    j1, j2 = f1.result(), f2.result()
    btc = j1.get("bitcoin", {})
    xrp = j2.get("ripple", {})
    return {
        "BTC": (btc.get("usd", np.nan), btc.get("usd_24h_change", np.nan)),
        "XRP": (xrp.get("btc", np.nan), None)
    }

# ── Load & Align Histories ──
btc_usd = load_hist_usd("bitcoin")
xrp_usd = load_hist_usd("ripple")
if btc_usd.empty or xrp_usd.empty:
    st.error("Failed to load historical data. Try again later.")
    st.stop()

idx      = btc_usd.index.intersection(xrp_usd.index)
btc_usd  = btc_usd.reindex(idx)
xrp_usd  = xrp_usd.reindex(idx)

xrp_btc         = pd.DataFrame(index=idx)
xrp_btc["price"]  = xrp_usd["price"] / btc_usd["price"]
xrp_btc["return"] = xrp_btc["price"].pct_change() * 100

# ── Compute Indicators ──
def compute_indicators(df):
    df["ema50"] = df["price"].ewm(span=EMA_T, adjust=False).mean()
    df["sma5"]  = df["price"].rolling(5).mean()
    df["sma20"] = df["price"].rolling(20).mean()
    df["vol14"] = df["return"].rolling(VOL_W).std().fillna(0)
    delta      = df["price"].diff()
    gain       = delta.clip(lower=0).rolling(RSI_W).mean()
    loss       = -delta.clip(upper=0).rolling(RSI_W).mean()
    df["rsi"]  = 100 - 100/(1 + gain/loss.replace(0, np.nan))
    return df.dropna()

btc_hist = compute_indicators(btc_usd.copy())
xrp_hist = compute_indicators(xrp_btc.copy())

# ── Live Prices ──
live         = load_live()
btc_p,btc_ch = live["BTC"]
xrp_p,_      = live["XRP"]

# ── ML Signal Generation & Training ──
def gen_sig(df, is_btc, params):
    X, y = [], []
    for i in range(EMA_T, len(df)-1):
        p, ret, vol = df["price"].iat[i], df["return"].iat[i], df["vol14"].iat[i]
        ed = p - df["ema50"].iat[i]
        mo = df["sma5"].iat[i] - df["sma20"].iat[i]
        rs = df["rsi"].iat[i]
        if is_btc:
            rsi_th, *_ = params
            cond = ed>0 and mo>0 and rs<rsi_th and ret>=vol
        else:
            m, b, _, dip = params
            mv = df["price"].rolling(m).mean().iat[i]
            cond = p<mv and ((mv-p)/p*100)>=dip and vol>df["vol14"].iat[i-1]
        if not cond:
            continue
        X.append([rs, vol, ed, mo, ret])
        y.append(1 if df["price"].iat[i+1] > p else 0)
    return np.array(X), np.array(y)

@st.cache_resource
def train_models(Xb, yb, Xx, yx):
    def build(X, y):
        if len(y)>=6 and len(np.unique(y))>1:
            gs = GridSearchCV(
                RandomForestClassifier(random_state=0),
                {"n_estimators":[50,100], "max_depth":[3,5]},
                cv=3, scoring="accuracy", n_jobs=1
            )
            gs.fit(X, y)
            return gs.best_estimator_
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        if len(y)>0:
            clf.fit(X, y)
        return clf
    return build(Xb, yb), build(Xx, yx)

def today_features(df):
    i = len(df)-1
    return [[
        df["rsi"].iat[i],
        df["vol14"].iat[i],
        df["price"].iat[i]-df["ema50"].iat[i],
        df["sma5"].iat[i]-df["sma20"].iat[i],
        df["return"].iat[i],
    ]]

def get_prob(clf, feat):
    if feat is None:
        return 0.0
    p = clf.predict_proba(feat)[0]
    return p[1] if len(p)>1 else 0.0

btc_params = (75, 1.5, 1.0)
xrp_params = (10, 75, 50, 1.0)
Xb, yb = gen_sig(btc_hist, True, btc_params)
Xx, yx = gen_sig(xrp_hist, False, xrp_params)
clf_b, clf_x = train_models(Xb, yb, Xx, yx)
p_b = get_prob(clf_b, today_features(btc_hist))
p_x = get_prob(clf_x, today_features(xrp_hist))

# ── Entry & Regime Logic ──
def regime_ok(df, prob):
    return (
        df["price"].iat[-1] > df["ema50"].iat[-1]
        and df["sma5"].iat[-1] > df["sma20"].iat[-1]
        and df["rsi"].iat[-1] < RSI_OB
        and df["vol14"].iat[-1] >= MIN_VOL
        and prob >= CLASS_THRESH
    )

def compute_drop(df, price, change):
    vol = df["vol14"].iat[-1]
    ret = change if change is not None else df["return"].iat[-1]
    if ret < vol or np.isnan(vol):
        return None
    return float(vol if ret <= 2*vol else 2*vol)

# ── Sidebar: Position Sizing & Manual Override ──
mode = st.sidebar.radio(
    "Mode",
    ("Start New Cycle", "Continue Existing"),
    index=0 if st.session_state.mode is None else (0 if st.session_state.mode=="new" else 1)
)
st.session_state.mode = "new" if mode=="Start New Cycle" else "cont"

usd_tot = st.sidebar.number_input("Total Investment ($)", 100.0, 1e6, 3000.0, 100.0)
pct_btc = st.sidebar.slider("BTC Allocation (%)", 0, 100, 70)
usd_btc = usd_tot * pct_btc / 100
gbp_rate = st.sidebar.number_input("GBP/USD Rate", 1.10, 1.60, 1.27, 0.01)
st.sidebar.metric("Portfolio Value", f"${usd_tot:,.2f}", f"£{usd_tot/gbp_rate:,.2f}")

min_ord = st.sidebar.number_input("Min Order (BTC)", 1e-6, 1e-2, 5e-4, 1e-6, format="%.6f")
MIN_O    = max(min_ord, (usd_btc/GRID_MAX)/btc_p if btc_p else 0)
st.sidebar.caption(f"Min Order ≥ {MIN_O:.6f} BTC (~${MIN_O*btc_p:.2f})")

override = st.sidebar.checkbox("Manual grid override", value=False)
if override:
    manual_b = st.sidebar.number_input("BTC/USDT grids", 2, GRID_MAX, 6)
    manual_x = st.sidebar.number_input("XRP/BTC grids", 2, GRID_MAX, 8)
else:
    manual_b = manual_x = None

# ── Automated State & Dynamic Grids ──
def auto_state(key, df, price, change, prob, low_c, up_c, cnt_c):
    # initial drop: use vol14 for first deploy, else dip logic
    if st.session_state.mode=="new" and not st.session_state[f"deployed_{key}"]:
        drop = df["vol14"].iat[-1]
    else:
        drop = compute_drop(df, price, change)

    deployed = st.session_state[f"deployed_{key}"]
    term     = st.session_state[f"terminated_{key}"]

    # Recover
    if term and regime_ok(df, prob):
        st.session_state[f"terminated_{key}"] = False
        term = False

    # Deploy
    if not deployed and not term and regime_ok(df, prob):
        st.session_state[f"deployed_{key}"] = True
        deployed = True

    # Bounds
    low = price*(1-drop/100) if (st.session_state.mode=="new" and drop is not None) else low_c
    up  = price if st.session_state.mode=="new" else up_c
    tp  = up*(1+drop/100) if (st.session_state.mode=="new" and drop is not None) else up_c

    # Recommended dynamic grids
    rec = max(5, min(GRID_MAX, int((usd_btc/price)//MIN_O or 0)))

    # Actual grid choice
    if st.session_state.mode=="cont":
        grids = cnt_c
    else:
        grids = manual_b if (override and key=="b") else manual_x if (override and key=="x") else rec

    # Action
    if deployed and price>=tp:
        st.session_state[f"terminated_{key}"] = True
        st.session_state[f"deployed_{key}"]   = False
        act = "Take-Profit"
    elif deployed and drop is not None and st.session_state.mode=="new":
        act = "Redeploy"
    elif term:
        act = "Terminated"
    elif not deployed:
        act = "Not Deployed"
    else:
        act = "Hold"

    return low, up, tp, grids, rec, act

# ── Render Each Bot ──
for key, label, hist, (pr, ch), prob in [
    ("b", "🟡 BTC/USDT", btc_hist, (btc_p, btc_ch), p_b),
    ("x", "🟣 XRP/BTC",   xrp_hist, (xrp_p,     None), p_x),
]:
    low, up, tp, actual_n, rec_n, act = auto_state(
        key, hist, pr, ch, prob,
        st.session_state.get(f"cont_low_{key}", pr),
        st.session_state.get(f"cont_up_{key}", pr),
        st.session_state.get(f"cont_grids_{key}", GRID_MAX),
    )

    st.subheader(f"{label} Bot")

    # hide metrics until deployable
    if act == "Not Deployed":
        st.info("⚠️ Waiting to deploy when conditions are met.")
        continue

    # once deployable or beyond, show side-by-side grids + prices
    c1, c2 = st.columns(2)
    if st.session_state.mode=="new":
        c1.metric("Grid Levels", f"{actual_n}")
        c2.metric("Recommended", f"{rec_n}")
    else:
        c1.metric("Grid Levels", f"{actual_n}")
        c2.write("")

    st.metric("Lower Price",    f"{low:,.6f}")
    st.metric("Upper Price",    f"{up:,.6f}")
    st.metric("Take-Profit At", f"{tp:,.6f}")

    if act=="Redeploy":
        st.info("🔔 Auto grid reset signal detected.")
    elif act=="Take-Profit":
        st.success("💰 TAKE-PROFIT executed—bot terminated.")
    elif act=="Terminated":
        st.error("🛑 Bot terminated—awaiting regime recovery.")
    else:
        st.info("⏸ HOLD—no action right now.")

# ── About & Features ──
with st.expander("ℹ️ About & Features"):
    st.markdown(
        "**Infinite Scalping Grid Bot Trading System** automates grid trading on Crypto.com:\n\n"
        "- **Live Signals**: EMA50, SMA crossover, RSI & volatility define deployable regimes.\n"
        "- **Machine Learning**: Random Forests trained on historical outcomes, requiring ≥80% win-prob to deploy.\n"
        "- **Automated Lifecycle**: Auto-deploy, auto-redeploy on dips, auto-take-profit, auto-recover.\n"
        "- **Flexible Grids**: Dynamic recommendations based on capital & min order, with manual override.\n\n"
        "**Symbols**:\n"
        "🟡 BTC/USDT, 🟣 XRP/BTC\n"
        "🔔 Redeploy, 💰 Take-Profit, 🛑 Terminated, ⏸ Hold"
    )

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
