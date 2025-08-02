import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from sklearn.ensemble import RandomForestRegressor
from streamlit_autorefresh import st_autorefresh
import time

# ── Auto‐refresh every 60 s ──
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
ML_THRESH    = 0.70
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
    js = fetch_json(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
                    {"vs_currency":vs,"days":days})
    prices = js.get("prices", [])
    df = pd.DataFrame(prices, columns=["ts","price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change()*100
    df["ema50"]  = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    df["sma5"]   = df["price"].rolling(5).mean()
    df["sma20"]  = df["price"].rolling(20).mean()
    df["vol14"]  = df["return"].rolling(VOL_WINDOW).std().fillna(0)
    d = df["price"].diff(); g=d.clip(lower=0); l=(-d).clip(upper=0)
    df["rsi"]   = 100-100/(1+g.rolling(RSI_WINDOW).mean()/l.rolling(RSI_WINDOW).mean())
    return df.dropna()

@st.cache_data(ttl=60)
def load_live():
    js = fetch_json("https://api.coingecko.com/api/v3/simple/price",
                    {"ids":"bitcoin,ripple","vs_currencies":"usd,btc","include_24hr_change":"true"})
    btc = js.get("bitcoin",{}); xrp=js.get("ripple",{})
    return {
      "BTC": (btc.get("usd",np.nan),btc.get("usd_24h_change",np.nan)),
      "XRP": (xrp.get("btc",np.nan),None)
    }

# ── Backtest Helpers ──
def backtest_btc(df,rsi,_,__):
    wins=loss=0
    for i in range(EMA_TREND,len(df)-1):
        p,ret,vol = df["price"].iat[i],df["return"].iat[i],df["vol14"].iat[i]
        cond = p>df["ema50"].iat[i] and df["sma5"].iat[i]>df["sma20"].iat[i] and df["rsi"].iat[i]<rsi
        if not cond or ret<vol: continue
        if df["price"].iat[i+1]>p: wins+=1
        else: loss+=1
    return wins/(wins+loss) if wins+loss else 0

def backtest_xrp(df,mean,bounce,_,dip):
    wins=loss=0
    df["mean"] = df["price"].rolling(mean).mean()
    df["vol"]  = df["return"].rolling(VOL_WINDOW).std().fillna(0)
    for i in range(mean,len(df)-1):
        p,m,vol=df["price"].iat[i],df["mean"].iat[i],df["vol"].iat[i]
        gap=(m-p)/p*100
        if not(p<m and gap>=dip and vol>df["vol"].iat[i-1]): continue
        if df["price"].iat[i+1]>=p+gap/100*p*bounce/100: wins+=1
        else: loss+=1
    return wins/(wins+loss) if wins+loss else 0

# ── Load Data ──
btc_hist = load_history("bitcoin","usd",HISTORY_DAYS)
xrp_hist = load_history("ripple","btc",HISTORY_DAYS)
live     = load_live()
btc_p,btc_ch = live["BTC"]
xrp_p,_     = live["XRP"]

# ── Build ML Training Sets ──
btc_rows=[]
for rsi in (65,70,75): 
    for tp in (1.0,1.5):
        for sl in (0.5,1.0):
            btc_rows.append((rsi,tp,sl, backtest_btc(btc_hist,rsi,tp,sl)))
btc_df=pd.DataFrame(btc_rows,columns=["rsi","tp","sl","win"])

xrp_rows=[]
for m in (5,10):
    for b in (50,75):
        for sl in (25,50):
            for d in (1.0,1.5):
                xrp_rows.append((m,b,sl,d, backtest_xrp(xrp_hist,m,b,sl,d)))
xrp_df=pd.DataFrame(xrp_rows,columns=["mean","bounce","sl","dip","win"])

from sklearn.ensemble import RandomForestRegressor
btc_ml=RandomForestRegressor(n_estimators=50, random_state=1)
btc_ml.fit(btc_df[["rsi","tp","sl"]],btc_df["win"])
xrp_ml=RandomForestRegressor(n_estimators=50, random_state=1)
xrp_ml.fit(xrp_df[["mean","bounce","sl","dip"]],xrp_df["win"])

# ── Determine Defaults & ML Override ──
btc_def=btc_df.loc[btc_df.win.idxmax(),["rsi","tp","sl"]].tolist()
xrp_def=xrp_df.loc[xrp_df.win.idxmax(),["mean","bounce","sl","dip"]].tolist()
btc_pred=btc_ml.predict([btc_def])[0]; ml_btc=btc_pred>=ML_THRESH
xrp_pred=xrp_ml.predict([xrp_def])[0]; ml_xrp=xrp_pred>=ML_THRESH

# ── Sidebar: Allocation & Custom Grids ──
st.sidebar.title("💰 Investment & Grids")
usd_total = st.sidebar.number_input("Total Investment ($)",100.0,1e6,3000.0,100.0)
split_pct = st.sidebar.slider("BTC Allocation (%)",0,100,70)
usd_btc = usd_total * split_pct/100
usd_xrp = usd_total - usd_btc
st.sidebar.metric("BTC Alloc.", f"${usd_btc:,.2f}", f"{split_pct}%")
st.sidebar.metric("XRP Alloc.", f"${usd_xrp:,.2f}", f"{100-split_pct}%")

gbp_rate = st.sidebar.number_input("GBP/USD Rate",1.1,1.6,1.27,0.01)
st.sidebar.caption(f"£{usd_total/gbp_rate:,.2f} total")

min_order=st.sidebar.number_input("Min Order (BTC)",1e-6,1e-2,5e-4,1e-6,format="%.6f")
MIN_ORDER=max(min_order,(usd_btc/GRID_MAX)/btc_p if btc_p else 0)
st.sidebar.caption(f"🔒 Min Order ≥ {MIN_ORDER:.6f} BTC")

custom_btc = st.sidebar.checkbox("Custom BTC Grid Levels",False)
if custom_btc:
    L_btc = st.sidebar.number_input("BTC Levels",1,GRID_MAX,GRID_PRIMARY,1)
else:
    L_btc = GRID_PRIMARY if not ml_btc else GRID_MAX

custom_xrp = st.sidebar.checkbox("Custom XRP Grid Levels",False)
if custom_xrp:
    L_xrp = st.sidebar.number_input("XRP Levels",1,GRID_MAX,GRID_PRIMARY,1)
else:
    L_xrp = GRID_PRIMARY if not ml_xrp else GRID_MAX

# ── Show Defaults ──
st.sidebar.markdown("### ⚙️ Final Defaults")
st.sidebar.write(f"BTC{' (ML)' if ml_btc else ''}: RSI<{int(btc_def[0])}, TP×{btc_def[1]}, SL{btc_def[2]}%")
st.sidebar.write(f"XRP{' (ML)' if ml_xrp else ''}: Mean{int(xrp_def[0])}d, Bounce{xrp_def[1]}%, SL{xrp_def[2]}%, Dip{xrp_def[3]}%")

# ── Compute & Display Grids ──
def show_bot(name,p,drop,levels,is_btc):
    st.header(name)
    unit = "USD" if is_btc else "BTC"
    st.write(f"- Price: {p:.6f} {unit}")
    if drop>0:
        if (is_btc and ml_btc) or (not is_btc and ml_xrp):
            st.success("✅ ML Override Active")
        st.write(f"- Drop: {drop:.2f}%")
        bot = p*(1-drop/100)
        step = (p-bot)/levels
        alloc = usd_btc if is_btc else usd_xrp
        per = (alloc/(p if is_btc else p*btc_p))/levels if not is_btc else alloc/p/levels
        st.write(f"- Lower: {bot:.6f}; Upper: {p:.6f}; Step: {step:.6f}")
        st.write(f"- Per Order: {per:.6f} BTC {'✅' if per>=MIN_ORDER else '❌'}")
        tp_price = p*(1+drop/100) if is_btc else p*(1+xrp_def[1]/100)
        st.write(f"- Take-Profit: {tp_price:.6f} {unit}")
        st.write("🔄 Redeploy Bot now")
    else:
        st.error("🛑 Terminate Bot")

vol14=btc_hist["vol14"].iat[-1]
ret24=btc_ch if btc_ch is not None else btc_hist["return"].iat[-1]
drop_btc=vol14 if ret24<vol14 else (2*vol14 if ret24>2*vol14 else ret24)
show_bot("🟡 BTC/USDT Bot", btc_p, drop_btc, L_btc, True)

sig_xrp = (xrp_hist["price"].iat[-1]<xrp_hist["price"].rolling(int(xrp_def[0])).mean().iat[-1]) \
          and (xrp_hist["vol14"].iat[-1]>xrp_hist["vol14"].iat[-2])
drop_xrp = xrp_def[1] if sig_xrp else 0
show_bot("🟣 XRP/BTC Bot", xrp_p, drop_xrp, L_xrp, False)

# ── About & Requirements ──
with st.expander("ℹ️ About"):
    st.markdown("""
    • Split your $ allocation between BTC and XRP bots.  
    • Optionally choose custom grid levels per bot.  
    • All original backtests, ML overrides, take-profit & deploy/terminate logic intact.
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
