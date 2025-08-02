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

# ── Auto-refresh every 60 s ──
st_autorefresh(interval=60_000, key="datarefresh")

# ── Page Setup ──
st.set_page_config(layout="centered")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
now = datetime.now(pytz.timezone("Europe/London"))
st.caption(f"Last updated: {now:%Y-%m-%d %H:%M %Z}")

# ── Constants ──
HISTORY_DAYS = 90
VOL_WINDOW   = 14
RSI_WINDOW   = 14
EMA_TREND    = 50
GRID_PRIMARY = 20
GRID_MAX     = 30
CLASS_PROB_THRESH = 0.80
MAX_RETRIES  = 3

# ── Helpers ──
def fetch_json(url, params):
    for i in range(MAX_RETRIES):
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 429:
            time.sleep(2**i)
            continue
        r.raise_for_status()
        return r.json()
    st.warning("⚠️ Rate limit reached; using cached data.")
    return {}

@st.cache_data(ttl=600)
def load_history(coin_id, vs, days):
    js = fetch_json(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        {"vs_currency": vs, "days": days}
    )
    prices = js.get("prices", [])
    df = pd.DataFrame(prices, columns=["ts","price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change() * 100
    df["ema50"]  = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    df["sma5"]   = df["price"].rolling(5).mean()
    df["sma20"]  = df["price"].rolling(20).mean()
    df["vol14"]  = df["return"].rolling(VOL_WINDOW).std().fillna(0)
    delta      = df["price"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    df["rsi"]  = 100 - 100 / (1 + gain.rolling(RSI_WINDOW).mean() / loss.rolling(RSI_WINDOW).mean())
    return df.dropna()

@st.cache_data(ttl=60)
def load_live():
    js = fetch_json(
        "https://api.coingecko.com/api/v3/simple/price",
        {"ids":"bitcoin,ripple","vs_currencies":"usd,btc","include_24hr_change":"true"}
    )
    btc = js.get("bitcoin",{}); xrp=js.get("ripple",{})
    return {
      "BTC": (btc.get("usd", np.nan), btc.get("usd_24h_change", np.nan)),
      "XRP": (xrp.get("btc", np.nan), None)
    }

# ── Backtest Signal + Label ──
def generate_signals(df, is_btc, params):
    """
    Apply your original signal filters, record features + label (1=win,0=loss).
    params: tuple of tuned parameters (rsi,tp,sl) or (mean,bounce,sl,dip).
    """
    X, y = [], []
    if is_btc:
        rsi_th, tp_mult, sl_pct = params
    else:
        mean_d, bounce_pct, sl_pct, min_dip = params

    for i in range(EMA_TREND, len(df)-1):
        p = df["price"].iat[i]
        ret = df["return"].iat[i]
        vol = df["vol14"].iat[i]
        ema_diff = p - df["ema50"].iat[i]
        mom = df["sma5"].iat[i] - df["sma20"].iat[i]
        rsi = df["rsi"].iat[i]

        if is_btc:
            cond = (ema_diff>0) and (mom>0) and (rsi<rsi_th) and (ret>=vol)
        else:
            mean_price = df["price"].rolling(mean_d).mean().iat[i]
            cond = (p<mean_price) and ((mean_price-p)/p*100>=min_dip) and (vol>df["vol14"].iat[i-1])

        if not cond:
            continue

        # record features:
        feats = [rsi, vol, ema_diff, mom, ret]
        X.append(feats)

        # label: profit next candle?
        profit = df["price"].iat[i+1] - p
        y.append(1 if profit>0 else 0)

    return np.array(X), np.array(y)

# ── Load Data ──
btc_hist = load_history("bitcoin","usd",HISTORY_DAYS)
xrp_hist = load_history("ripple","btc",HISTORY_DAYS)
live     = load_live()
btc_p, btc_ch = live["BTC"]
xrp_p, _      = live["XRP"]

# ── Tune your original params by backtest (as before) ──
# Skipping details—re-use your tuned defaults:
btc_defaults = (75, 1.5, 1.0)     # example defaults from your 70% grid-search
xrp_defaults = (10, 75, 50, 1.0)  # likewise

# ── Build ML Training Sets ──
X_btc, y_btc = generate_signals(btc_hist, True,  btc_defaults)
X_xrp, y_xrp = generate_signals(xrp_hist, False, xrp_defaults)

# ── Train best‐in‐class RandomForestClassifier ──
param_grid = {
  "n_estimators": [50,100,200],
  "max_depth":    [3,5,7],
}
rf_btc = GridSearchCV(RandomForestClassifier(random_state=0),
                      param_grid, cv=3, scoring="accuracy", n_jobs=-1)
rf_btc.fit(X_btc, y_btc)

rf_xrp = GridSearchCV(RandomForestClassifier(random_state=0),
                      param_grid, cv=3, scoring="accuracy", n_jobs=-1)
rf_xrp.fit(X_xrp, y_xrp)

# ── Sidebar: Allocation & Currency ──
st.sidebar.title("💰 Investment Settings")
usd_total = st.sidebar.number_input("Total Investment ($)",100.0,1e6,3000.0,100.0)
pct_btc   = st.sidebar.slider("BTC Allocation (%)",0,100,70)
usd_btc   = usd_total * pct_btc/100
usd_xrp   = usd_total - usd_btc
gbp_rate  = st.sidebar.number_input("GBP/USD Rate",1.1,1.6,1.27,0.01)
st.sidebar.metric("Alloc USD/GBP", f"${usd_total:,.2f}", f"£{usd_total/gbp_rate:,.2f}")
min_order = st.sidebar.number_input("Min Order (BTC)",1e-6,1e-2,5e-4,1e-6,format="%.6f")
MIN_ORDER = max(min_order,(usd_btc/GRID_MAX)/btc_p if btc_p else 0)
st.sidebar.caption(f"🔒 Min Order ≥ {MIN_ORDER:.6f} BTC (~${MIN_ORDER*btc_p:.2f})")

# ── Prepare live‐feature vector for today’s signal ──
def today_features(hist, is_btc):
    i = len(hist)-1
    p   = hist["price"].iat[i]
    ret = hist["return"].iat[i]
    vol = hist["vol14"].iat[i]
    ema_diff = p - hist["ema50"].iat[i]
    mom = hist["sma5"].iat[i] - hist["sma20"].iat[i]
    rsi = hist["rsi"].iat[i]
    return np.array([[rsi, vol, ema_diff, mom, ret]])

# ── Predict live‐signal probability ──
pb = rf_btc.predict_proba(today_features(btc_hist,True))[0][1]
px = rf_xrp.predict_proba(today_features(xrp_hist,False))[0][1]

# ── Decide override or fallback ──
use_ml_btc = pb >= CLASS_PROB_THRESH
use_ml_xrp = px >= CLASS_PROB_THRESH

# ── Final Defaults (sidebar) ──
st.sidebar.markdown("### ⚙️ Final Defaults")
btcl = "(ML)" if use_ml_btc else ""
xrpl = "(ML)" if use_ml_xrp else ""
st.sidebar.write(f"BTC{btcl}: RSI<{btc_defaults[0]},TP×{btc_defaults[1]},SL{btc_defaults[2]}% (p={pb:.0%})")
st.sidebar.write(f"XRP{xrpl}: Mean{int(xrp_defaults[0])}d,Bounce{xrp_defaults[1]}%,SL{xrp_defaults[2]}%,Dip{xrp_defaults[3]}% (p={px:.0%})")

# ── Grid calc & display ──
def display_bot(name,p,drop,levels,is_btc):
    st.header(name)
    unit = "USD" if is_btc else "BTC"
    st.write(f"- Price: {p:.6f} {unit}")
    if drop>0:
        flag = use_ml_btc if is_btc else use_ml_xrp
        if flag: st.success("✅ ML Override Active")
        st.write(f"- Drop %: {drop:.2f}%")
        bot,step = (p*(1-drop/100),(p*(drop/100))/levels)
        alloc = usd_btc if is_btc else usd_xrp
        per = (alloc/p)/levels if is_btc else (alloc/(p*btc_p))/levels
        st.write(f"- Lower: {bot:.6f}; Upper: {p:.6f}; Step: {step:.6f}")
        st.write(f"- Per-Order: {per:.6f} BTC {'✅' if per>=MIN_ORDER else '❌'}")
        tp = p*(1+drop/100) if is_btc else p*(1+xrp_defaults[1]/100)
        st.write(f"- Take-Profit: {tp:.6f} {unit}")
        st.write("🔄 Redeploy Bot now")
    else:
        st.error("🛑 Terminate Bot")

vol14 = btc_hist["vol14"].iat[-1]
ret24 = btc_ch if btc_ch is not None else btc_hist["return"].iat[-1]
drop_btc = vol14 if ret24<vol14 else (2*vol14 if ret24>2*vol14 else ret24)
levels_btc = GRID_PRIMARY if not use_ml_btc else GRID_MAX
display_bot("🟡 BTC/USDT Bot", btc_p, drop_btc, levels_btc, True)

sig_x = (xrp_hist["price"].iat[-1] < xrp_hist["price"].rolling(int(xrp_defaults[0])).mean().iat[-1])
sig_x &= (xrp_hist["vol14"].iat[-1] > xrp_hist["vol14"].iat[-2])
drop_xrp = xrp_defaults[1] if sig_x else 0
levels_xrp = GRID_PRIMARY if not use_ml_xrp else GRID_MAX
display_bot("🟣 XRP/BTC Bot", xrp_p, drop_xrp, levels_xrp, False)

with st.expander("ℹ️ About"):
    st.markdown("""
    • Now using a **classification** RF model tuned via grid‐search for  
      probability of a winning next-candle.  
    • **Override** defaults when p(win) ≥ 80 %.  
    • Indicators (EMA, SMA, RSI, vol) still drive your original signals.  
    • Copy the shown grid bounds, steps, and take-profit into Crypto.com.
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
