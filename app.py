import streamlit as st
import requests
import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh

# Auto‑refresh
st_autorefresh(interval=60_000, key="datarefresh")

# Config
HISTORY_DAYS   = 90
VOL_WINDOW     = 14
RSI_WINDOW     = 14
SMA_SHORT      = 5
SMA_LONG       = 20
EMA_TREND      = 50
RSI_OVERBOUGHT = 75

# Grid‐level bounds
GRID_MIN, GRID_MAX = 1, 30

# Fetch 90 d BTC/USD history + indicators
@st.cache_data(ttl=600)
def fetch_history(days):
    r = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
        params={"vs_currency":"usd","days":days}, timeout=10
    ); r.raise_for_status()
    df = pd.DataFrame(r.json()["prices"], columns=["ts","price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    df["return"] = df["price"].pct_change()*100
    df["vol14"]  = df["return"].rolling(VOL_WINDOW).std()
    df["sma5"]   = df["price"].rolling(SMA_SHORT).mean()
    df["sma20"]  = df["price"].rolling(SMA_LONG).mean()
    df["ema50"]  = df["price"].ewm(span=EMA_TREND, adjust=False).mean()
    d = df["price"].diff()
    g, l = d.clip(lower=0), -d.clip(upper=0)
    df["rsi"] = 100 - 100/(1 + g.rolling(RSI_WINDOW).mean()/l.rolling(RSI_WINDOW).mean())
    return df.dropna()

# Fetch live prices
@st.cache_data(ttl=60)
def fetch_live():
    r = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={
          "ids":"bitcoin,ripple",
          "vs_currencies":"usd,btc",
          "include_24hr_change":"true"
        }, timeout=10
    ); r.raise_for_status()
    j = r.json()
    return {
      "BTC/USDT": (j["bitcoin"]["usd"], j["bitcoin"]["usd_24h_change"]),
      "XRP/BTC": (j["ripple"]["btc"], None)
    }

# Compute grid
def compute_grid(top, drop_pct, levels):
    bottom = top*(1 - drop_pct/100)
    step = (top - bottom)/levels
    return bottom, step

# Inputs
st.sidebar.title("💰 Investment Settings")
inv_btc   = st.sidebar.number_input("Total Investment (BTC)", 1e-5, 1.0, 0.01, 1e-5, "%.5f")
min_order = st.sidebar.number_input("Min Order Size (BTC)", 1e-6, 0.01, 5e-4, 1e-6, "%.6f")

# Data
hist   = fetch_history(HISTORY_DAYS)
live   = fetch_live()
latest = hist.iloc[-1]
vol14  = latest["vol14"]

# Page
st.set_page_config(layout="centered")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")

# Shared backtest of the trigger on BTC/USD
# Identify signal days & wins
mod_th = vol14
str_th = 2*vol14
cond = (hist["return"]>=mod_th)&(hist["price"]>hist["ema50"])&(hist["sma5"]>hist["sma20"])&(hist["rsi"]<RSI_OVERBOUGHT)
trades = cond.sum()
wins   = ((hist["price"].shift(-1) > hist["price"]) & cond).sum()
win_rate = wins/trades if trades else 0

# Optimize grid count for BTC/USDT
drop_pct_btc = (live["BTC/USDT"][1] if live["BTC/USDT"][1]>=mod_th else
                mod_th if live["BTC/USDT"][1]<=str_th else str_th)
scores = []
for L in range(GRID_MIN, GRID_MAX+1):
    # expected gain per signal = win_rate * (drop_pct_btc / L)
    scores.append(win_rate*(drop_pct_btc/L))
opt_L = int(np.argmax(scores)+GRID_MIN)
# define fewer/more around opt_L
few_L = max(GRID_MIN, opt_L-10)
mor_L = min(GRID_MAX, opt_L+10)

def run_bot(name, pair, top_price, pct_change):
    """Render a bot section for given pair"""
    st.header(f"{name} ({pair})")
    st.write(f"- Price: {top_price:.8f}")
    if pct_change is not None:
        st.write(f"- 24 h Change: {pct_change:.2f}%")
    st.write(f"- 14 d Vol: {vol14:.2f}%")
    filters_ok = (latest["price"]>latest["ema50"]) and (latest["sma5"]>latest["sma20"]) and (latest["rsi"]<RSI_OVERBOUGHT)
    st.write(f"- Filters: Regime={'✅' if latest['price']>latest['ema50'] else '❌'}, "
             f"Momentum={'✅' if latest['sma5']>latest['sma20'] else '❌'}, "
             f"RSI={'✅' if latest['rsi']<RSI_OVERBOUGHT else '❌'}")

    change = pct_change if pct_change is not None else hist["return"].iloc[-1]
    # determine reset
    if change < mod_th:
        drop, status = None, f"No reset ({change:.2f}% < {mod_th:.2f}%)"
    elif change <= str_th:
        drop, status = mod_th,  f"🔔 Moderate reset → drop {mod_th:.2f}%"
    else:
        drop, status = str_th,  f"🔔 Strong reset → drop {str_th:.2f}%"
    st.markdown(f"**Status:** {status}")

    if drop is not None and filters_ok:
        st.subheader("📈 Grid Recommendations")
        # pick L based on pair
        if pair=="BTC/USDT":
            primary, few, more = opt_L, few_L, mor_L
        else:
            primary, few, more = GRID_PRIMARY, GRID_FEWER, GRID_MORE

        for L,label in [(primary,"Most Profitable"),(few,"Fewer"),(more,"More")]:
            bottom,step = compute_grid(top_price, drop, L)
            per = inv_btc/L
            valid = per>=min_order
            st.markdown(f"**{label} ({L} levels)**  \n"
                        f"- Lower: `{bottom:.8f}`\n"
                        f"- Upper: `{top_price:.8f}`\n"
                        f"- Step: `{step:.8f}`\n"
                        f"- Per‐Order: `{per:.6f}` BTC {'✅' if valid else '❌'}")
        # full table
        st.write("### Grid vs. Per‐Order Table")
        tbl=[]
        for L in range(GRID_MIN,GRID_MAX+1):
            _,step=compute_grid(top_price,drop,L)
            per=inv_btc/L
            tbl.append({"Levels":L,"Step":f"{step:.8f}","Per‐Order":f"{per:.6f}",
                        "Valid?":"✅" if per>=min_order else "❌"})
        st.table(pd.DataFrame(tbl))
    else:
        st.info("No grid adjustment at this time.")

# Run bots
run_bot("XRP/BTC Bot","XRP/BTC", *live["XRP/BTC"])
run_bot("BTC/USDT Bot","BTC/USDT",*live["BTC/USDT"])

# Backtest summary
st.subheader("⚙️ Strategy Backtest (BTC/USD Signals over 90 d)")
st.write(f"- Signals: {trades} | Wins: {wins} | Win Rate: {win_rate*100:.1f}%")
