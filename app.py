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

# ── Persisted Flags ──
for b in ("b","x"):
    st.session_state.setdefault(f"deployed_{b}", False)
    st.session_state.setdefault(f"terminated_{b}", False)
st.session_state.setdefault("mode", None)
for b in ("b","x"):
    st.session_state.setdefault(f"cont_low_{b}", None)
    st.session_state.setdefault(f"cont_up_{b}", None)
    st.session_state.setdefault(f"cont_grids_{b}", None)

# ── Constants ──
H_DAYS, VOL_W, RSI_W, EMA_T = 90, 14, 14, 50
RSI_OB, MIN_VOL = 75, 1.0
GRID_DEF, GRID_MAX = 20, 30
CLASS_THRESH = 0.80
MAX_R = 3

# ── Fetch + Cache ──
def fetch_json(url, params):
    for i in range(MAX_R):
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 429:
            time.sleep(2**i)
            continue
        r.raise_for_status()
        return r.json()
    return {}

@st.cache_data(ttl=600)
def load_hist(coin, vs):
    js = fetch_json(
        f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
        {"vs_currency":vs,"days":H_DAYS}
    ) or {}
    df = pd.DataFrame(js.get("prices", []), columns=["ts","price"])
    if df.empty: return df
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change() * 100
    df["ema50"]  = df["price"].ewm(span=EMA_T, adjust=False).mean()
    df["sma5"]   = df["price"].rolling(5).mean()
    df["sma20"]  = df["price"].rolling(20).mean()
    df["vol14"]  = df["return"].rolling(VOL_W).std()
    d = df["price"].diff()
    df["rsi"]    = 100 - 100 / (
        1
        + df["price"].diff().clip(lower=0).rolling(RSI_W).mean()
        / (-df["price"].diff().clip(upper=0)).rolling(RSI_W).mean()
    )
    return df

@st.cache_data(ttl=60)
def load_live():
    def one(id,vs,ext):
        return fetch_json(
            "https://api.coingecko.com/api/v3/simple/price",
            {"ids":id,"vs_currencies":vs,**ext}
        ) or {}
    with concurrent.futures.ThreadPoolExecutor() as ex:
        b = ex.submit(one, "bitcoin", "usd", {"include_24hr_change":"true"})
        x = ex.submit(one, "ripple",  "btc", {})
    j1,j2 = b.result(), x.result()
    return {
        "BTC": (j1.get("bitcoin",{}).get("usd", np.nan),
                j1.get("bitcoin",{}).get("usd_24h_change", np.nan)),
        "XRP": (j2.get("ripple",{}).get("btc", np.nan), None)
    }

# ── Data Load ──
btc_hist = load_hist("bitcoin","usd")
xrp_hist = load_hist("ripple","btc")
live     = load_live()
btc_p,btc_ch = live["BTC"]
xrp_p,_     = live["XRP"]

# ── Regime & Drop Logic ──
def regime_ok(df):
    if df.empty: return False
    return (
        df["price"].iat[-1] > df["ema50"].iat[-1]
        and df["sma5"].iat[-1] > df["sma20"].iat[-1]
        and df["rsi"].iat[-1] < RSI_OB
        and df["vol14"].iat[-1] >= MIN_VOL
    )

def compute_drop(df, price, change):
    if df.empty: return 0
    vol = df["vol14"].iat[-1]
    ret = change if change is not None else df["return"].iat[-1]
    if np.isnan(vol) or ret < vol:
        return 0
    return vol if ret <= 2*vol else 2*vol

# ── ML Prep ──
def gen_sig(df, is_btc, ps):
    X,y = [],[]
    for i in range(EMA_T, len(df)-1):
        p,ret,vol = df["price"].iat[i], df["return"].iat[i], df["vol14"].iat[i]
        ed = p - df["ema50"].iat[i]
        mo = df["sma5"].iat[i] - df["sma20"].iat[i]
        rs = df["rsi"].iat[i]
        if is_btc:
            rsi_th,*_ = ps
            cond = ed>0 and mo>0 and rs<rsi_th and ret>=vol
        else:
            m,b,_,dip = ps
            mv = df["price"].rolling(m).mean().iat[i]
            cond = p<mv and ((mv-p)/p*100)>=dip and vol > df["vol14"].iat[i-1]
        if not cond: continue
        X.append([rs, vol, ed, mo, ret])
        y.append(1 if df["price"].iat[i+1] > p else 0)
    return np.array(X), np.array(y)

@st.cache_resource
def train(Xb,yb,Xx,yx):
    def t(X,y):
        if len(y)>=6 and len(np.unique(y))>1:
            gs = GridSearchCV(
                RandomForestClassifier(random_state=0),
                {"n_estimators":[50,100],"max_depth":[3,5]},
                cv=3, scoring="accuracy", n_jobs=1
            )
            gs.fit(X,y)
            return gs.best_estimator_
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        if len(y)>0:
            clf.fit(X,y)
        return clf
    return t(Xb,yb), t(Xx,yx)

def today_feat(df):
    if df.empty:
        return None
    i = len(df)-1
    return [[
        df["rsi"].iat[i],
        df["vol14"].iat[i],
        df["price"].iat[i] - df["ema50"].iat[i],
        df["sma5"].iat[i] - df["sma20"].iat[i],
        df["return"].iat[i],
    ]]

def safe_prob(clf, feat):
    if feat is None:
        return 0.0
    p = clf.predict_proba(feat)[0]
    return p[1] if len(p)>1 else 0.0

# ── Train ML ──
btc_ps = (75,1.5,1.0)
xrp_ps = (10,75,50,1.0)
Xb,yb = gen_sig(btc_hist, True,  btc_ps)
Xx,yx = gen_sig(xrp_hist, False, xrp_ps)
clf_b,clf_x = train(Xb,yb,Xx,yx)
p_b = safe_prob(clf_b, today_feat(btc_hist))
p_x = safe_prob(clf_x, today_feat(xrp_hist))

# ── Sidebar: Mode ──
mode = st.sidebar.radio(
    "Mode",
    ("Start New Cycle","Continue Existing"),
    index=0 if st.session_state.mode is None else (0 if st.session_state.mode=="new" else 1)
)
st.session_state.mode = "new" if mode=="Start New Cycle" else "cont"

# ── Sidebar: Allocation & Grids ──
usd_tot = st.sidebar.number_input("Total Investment ($)",100.0,1e6,3000.0,100.0)
pct_btc = st.sidebar.slider("BTC Allocation (%)",0,100,70)
usd_btc, usd_xrp = usd_tot*pct_btc/100, usd_tot - usd_tot*pct_btc/100
gbp_rate = st.sidebar.number_input("GBP/USD Rate",1.10,1.60,1.27,0.01)
st.sidebar.metric("Portfolio", f"${usd_tot:,.2f}", f"£{usd_tot/gbp_rate:,.2f}")
min_ord = st.sidebar.number_input("Min Order (BTC)",1e-6,1e-2,5e-4,1e-6,format="%.6f")
MIN_O = max(min_ord, (usd_btc/GRID_MAX)/btc_p if btc_p else 0)
st.sidebar.caption(f"Min Order ≥ {MIN_O:.6f} BTC (~${MIN_O*btc_p:.2f})")

default_b = GRID_MAX if p_b>=CLASS_THRESH else GRID_DEF
default_x = GRID_MAX if p_x>=CLASS_THRESH else GRID_DEF
g_b = st.sidebar.slider("BTC Grid Levels",5,GRID_MAX,default_b,key="gb")
g_x = st.sidebar.slider("XRP Grid Levels",5,GRID_MAX,default_x,key="gx")

# ── Continue mode inputs ──
if st.session_state.mode=="cont":
    st.sidebar.markdown("### Existing Bot Configs")
    for b,name,p in (("b","BTC/USDT",btc_p),("x","XRP/BTC",xrp_p)):
        low = st.sidebar.number_input(
            f"{name} Lower", 0.0, 10*p,
            value=st.session_state[f"cont_low_{b}"] or p,
            format="%.6f"
        )
        up = st.sidebar.number_input(
            f"{name} Upper", low, 10*p,
            value=st.session_state[f"cont_up_{b}"] or p,
            format="%.6f"
        )
        cnt = st.sidebar.slider(
            f"{name} Grid Count", 5, GRID_MAX,
            value=st.session_state[f"cont_grids_{b}"] or GRID_DEF
        )
        st.session_state[f"cont_low_{b}"]=low
        st.session_state[f"cont_up_{b}"]=up
        st.session_state[f"cont_grids_{b}"]=cnt

# ── Decision Logic ──
def decide(df, price, chg, cl, cu, cg, key):
    drop = compute_drop(df, price, chg)
    low = price*(1-drop/100) if st.session_state.mode=="new" else cl
    up  = price if st.session_state.mode=="new" else cu
    tp  = up*(1+drop/100) if st.session_state.mode=="new" else up
    grids = (
        GRID_MAX if (p_b if key=="b" else p_x)>=CLASS_THRESH else GRID_DEF
    ) if (st.session_state.mode=="new" and drop>0) else (
        cg if st.session_state.mode=="cont" else GRID_DEF
    )

    if st.session_state[f"terminated_{key}"]:
        if regime_ok(df):
            st.session_state[f"terminated_{key}"] = False
            return low,up,tp,grids,"Not Deployed"
        return low,up,tp,grids,"Terminated"

    if not regime_ok(df):
        return low,up,tp,grids,"Terminate"
    if not st.session_state[f"deployed_{key}"]:
        return low,up,tp,grids,"Not Deployed"
    if st.session_state.mode=="new" and drop>0:
        return low,up,tp,grids,"Redeploy"
    if price>=tp:
        return low,up,tp,grids,"Take-Profit"
    return low,up,tp,grids,"Hold"

# ── Render Bot Cards ──
for b,name,hist,(pr,ch) in [
    ("b","🟡 BTC/USDT", btc_hist,(btc_p,btc_ch)),
    ("x","🟣 XRP/BTC",  xrp_hist,(xrp_p,None))
]:
    low,up,tp,gd,act = decide(
        hist, pr, ch,
        st.session_state[f"cont_low_{b}"],
        st.session_state[f"cont_up_{b}"],
        st.session_state[f"cont_grids_{b}"],
        b
    )
    st.subheader(name+" Bot")
    st.metric("Grids",       f"{gd}")
    st.metric("Lower Price", f"{low:,.6f}")
    st.metric("Upper Price", f"{up:,.6f}")
    st.metric("Take-Profit", f"{tp:,.6f}")

    if act=="Terminated":
        st.error("🛑 Terminated—waiting for regime to recover.")
        continue
    if act=="Terminate":
        if st.button("🛑 Terminate Bot", key=f"{b}_term"):
            st.session_state[f"terminated_{b}"] = True
        continue
    if act=="Not Deployed":
        st.warning("⚠️ Not Deployed—click Deploy to start.")
        if st.button("🔄 Deploy", key=f"{b}_dep"):
            st.session_state[f"deployed_{b}"] = True
        continue
    if act=="Redeploy":
        st.info("🔔 Grid Reset Signal")
        if st.button("🔄 Redeploy Now", key=f"{b}_red"):
            st.success("✅ Copy to Crypto.com Grid Bot")
        continue
    if act=="Take-Profit":
        st.success("💰 TAKE-PROFIT: close & terminate bot on exchange")
        continue
    if act=="Hold":
        st.info("⏸ HOLD—no action right now.")
        continue

# ── Help & Requirements ──
with st.expander("ℹ️ How to Use"):
    st.write("""
    **Start New Cycle**  
      1. Wait for “Not Deployed” → **Deploy** when regime OK.  
      2. **Redeploy** on dips → copy new bounds.  
      3. **Take-Profit** at upper → close & terminate.  

    **Continue Existing Cycle**  
      1. Enter your current Lower, Upper & Grid Count.  
      2. App immediately shows you Hold/Redeploy/Take-Profit/Terminate.  

    On **Terminate**, liquidate on exchange, then the app waits until EMA50, SMA-crossover, RSI<75 and vol≥1% are all OK before letting you Deploy again.
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
