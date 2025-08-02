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
HISTORY_DAYS   = 90
VOL_WINDOW     = 14
RSI_WINDOW     = 14
EMA_TREND      = 50
GRID_PRIMARY   = 20
GRID_FEWER     = 10
GRID_MORE      = 30
GRID_MAX       = 30
ML_THRESH      = 0.70   # ML predicted win‐rate threshold
MAX_RETRIES    = 3      # for CoinGecko retry

# ── Helpers ──
def fetch_json(url, params):
    for i in range(MAX_RETRIES):
        r = requests.get(url, params=params, timeout=10)
        if r.status_code==429:
            time.sleep(2**i)
            continue
        r.raise_for_status()
        return r.json()
    st.warning("⚠️ Rate limit reached; using stale/cached data.")
    return {}

@st.cache_data(ttl=600)
def load_history(coin_id, vs, days):
    js = fetch_json(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
                    {"vs_currency":vs,"days":days})
    prices = js.get("prices",[])
    df = pd.DataFrame(prices, columns=["ts","price"])
    df["date"] = pd.to_datetime(df["ts"],unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change()*100
    df["ema50"]  = df["price"].ewm(span=EMA_TREND,adjust=False).mean()
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

# ── Backtest Functions ──
def backtest_btc(df,rsi,_,__):
    return backtest_generic(df, "BTC")

def backtest_xrp(df,_,__,___,____):
    return backtest_generic(df, "XRP")

def backtest_generic(df,pair):
    wins=loss=0
    for i in range(EMA_TREND,len(df)-1):
        p,ret,vol = df["price"].iat[i],df["return"].iat[i],df["vol14"].iat[i]
        if pair=="BTC":
            cond = p>df["ema50"].iat[i] and df["sma5"].iat[i]>df["sma20"].iat[i] and df["rsi"].iat[i]<75
        else:
            m=df["price"].rolling(10).mean().iat[i]
            cond = p<m and df["vol14"].iat[i]>df["vol14"].iat[i-1]
        if not cond: continue
        profit = df["price"].iat[i+1]-p
        if profit>0: wins+=1
        else: loss+=1
    return wins/(wins+loss) if wins+loss else 0

# ── Data Load ──
btc_hist = load_history("bitcoin","usd",HISTORY_DAYS)
xrp_hist = load_history("ripple","btc",HISTORY_DAYS)
live     = load_live()
btc_p,btc_ch = live["BTC"]
xrp_p,_     = live["XRP"]

# ── Build ML Training Sets ──
btc_rows=[]  
for rsi,tp,sl in [(r, t, s) for r in (65,70,75) for t in (1,1.5) for s in (0.5,1)]:
    btc_rows.append((rsi,tp,sl, backtest_btc(btc_hist,rsi,tp,sl)))
btc_df = pd.DataFrame(btc_rows,columns=["rsi","tp","sl","win"])

xrp_rows=[]
for m,b,sl,d in [(m,b,s,d) for m in (5,10) for b in (50,75) for s in (25,50) for d in (1,1.5)]:
    xrp_rows.append((m,b,sl,d, backtest_xrp(xrp_hist,m,b,sl,d)))
xrp_df = pd.DataFrame(xrp_rows,columns=["mean","bounce","sl","dip","win"])

from sklearn.ensemble import RandomForestRegressor
btc_ml=RandomForestRegressor(50,random_state=1)
xrp_ml=RandomForestRegressor(50,random_state=1)
btc_ml.fit(btc_df[["rsi","tp","sl"]],btc_df["win"])
xrp_ml.fit(xrp_df[["mean","bounce","sl","dip"]],xrp_df["win"])

# ── Determine Defaults & ML Override ──
btc_def = btc_df.loc[btc_df.win.idxmax(),["rsi","tp","sl"]]
xrp_def = xrp_df.loc[xrp_df.win.idxmax(),["mean","bounce","sl","dip"]]
btc_pred=btc_ml.predict([btc_def])[0]; ml_btc=btc_pred>=ML_THRESH
xrp_pred=xrp_ml.predict([xrp_def])[0]; ml_xrp=xrp_pred>=ML_THRESH

# ── Sidebar: Conversion & Settings ──
st.sidebar.title("💰 Settings")
usd_alloc = st.sidebar.number_input("Investment ($)",10.0,1e6,500.0,10.0)
gbp_rate  = st.sidebar.number_input("GBP/USD",1.1,1.5,1.27,0.01)
st.sidebar.metric("Value",f"${usd_alloc:,.2f}",f"£{usd_alloc/gbp_rate:,.2f}")
min_order = st.sidebar.number_input("Min Order (BTC)",1e-6,1e-2,5e-4,1e-6)
MIN_ORDER = max(min_order,(usd_alloc/GRID_MAX)/btc_p if btc_p else 0)

# ── Display Final Params ──
st.sidebar.markdown("### ⚙️ Params")
st.sidebar.write(f"BTC {'(ML)' if ml_btc else ''}: RSI<{int(btc_def.rsi)},TP×{btc_def.tp},SL{btc_def.sl}%")
st.sidebar.write(f"XRP {'(ML)' if ml_xrp else ''}: Mean{int(xrp_def.mean)}d,Bounce{xrp_def.bounce}%,SL{xrp_def.sl}%,Dip{xrp_def.dip}%")

# ── Grid Helper ──
def grid_ui(name,p,drop,levels,pair):
    st.header(name)
    st.write(f"- Price: {p:.6f}" + (" BTC" if pair=="XRP" else " USD"))
    if drop>0:
        if (pair=="BTC" and ml_btc) or (pair=="XRP" and ml_xrp):
            st.success("✅ ML Override Active")
        st.write(f"- Drop %: {drop:.2f}%")
        bot,step=(p*(1-drop/100),(p*(drop/100))/levels)
        per=(usd_alloc/p)/levels
        st.write(f"- Lower: {bot:.6f}; Upper: {p:.6f}; Step: {step:.6f}")
        st.write(f"- Per-Order: {per:.6f} BTC {'✅' if per>=MIN_ORDER else '❌'}")
        st.write("- Take-Profit:" + ("  >> $" + f"{p*(1+drop/100):.2f}" if pair=="BTC" else f"  >> {p*(1+ xrp_def.bounce/100):.6f} BTC"))
        st.write("🔄 Redeploy Bot now")
    else:
        st.error("🛑 Terminate Bot")

# ── Show BTC & XRP Grids ──
vol14=btc_hist.vol14.iat[-1]; ret24=btc_ch if btc_ch else btc_hist.return.iat[-1]
drop_btc = vol14 if ret24<vol14 else (2*vol14 if ret24>2*vol14 else ret24)
grid_ui("🟡 BTC/USDT Bot",btc_p,drop_btc, GRID_MORE if ml_btc else GRID_PRIMARY, "BTC")

sig_xrp=(xrp_hist.price.iat[-1]<xrp_hist.price.rolling(int(xrp_def.mean)).mean().iat[-1]) \
        and(xrp_hist.vol14.iat[-1]>xrp_hist.vol14.iat[-2])
drop_xrp=xrp_def.bounce if sig_xrp else 0
grid_ui("🟣 XRP/BTC Bot",xrp_p,drop_xrp, GRID_MORE if ml_xrp else GRID_PRIMARY, "XRP")

with st.expander("ℹ️ About"):
    st.write("• BTC uses trend-pullback; XRP uses mean-reversion.")
    st.write("• ML override when predicted win-rate ≥70 %.")
    st.write("• Take-profit level shown; terminate/redeploy guidance provided.")
