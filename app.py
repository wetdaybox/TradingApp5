# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# ── Auto-refresh every minute ──
st_autorefresh(interval=60_000, key="refresh")

# ── Page setup ──
st.set_page_config(layout="centered")
st.title("🇬🇧 Infinite Scalping Grid Bot Trading System")
now = datetime.now(pytz.timezone("Europe/London"))
st.caption(f"Last updated: {now:%Y-%m-%d %H:%M %Z}")

# ── Investment inputs ──
usd_alloc      = st.sidebar.number_input("Total Investment ($)", 10.0, 1e6, 500.0, 10.0)
user_min_order = st.sidebar.number_input("Min Order (BTC)",   1e-6,1e-2,5e-4,1e-6,format="%.6f")

# ── Constants ──
VOL_WINDOW = ATR_WINDOW = ADX_WINDOW = RSI_WINDOW = 14
GRID_PRIMARY, GRID_FEWER, GRID_MORE, GRID_MAX = 20,10,30,30

# ── Fetch 1 yr history via yfinance ──
@st.cache_data(ttl=3600)
def fetch_history(ticker):
    df = yf.download(ticker, period="1y", interval="1d")[["Close"]]
    df = df.rename(columns={"Close":"price"}).dropna()
    df["return"] = df["price"].pct_change()*100
    return df.dropna()

btc_df = fetch_history("BTC-USD")
xrp_df = fetch_history("XRP-BTC")

# ── Derive indicators & simulate grid outcomes ──
def engineer_and_label(df, is_btc=True):
    df = df.copy().reset_index(drop=True)
    # Indicators
    df["ATR"] = df["return"].abs().rolling(ATR_WINDOW).mean()
    up   = df["price"].diff().clip(lower=0).rolling(ADX_WINDOW).mean()
    down = -df["price"].diff().clip(upper=0).rolling(ADX_WINDOW).mean()
    df["ADX"] = (abs(up-down)/(up+down)).rolling(ADX_WINDOW).mean()*100
    delta = df["price"].diff()
    gain  = delta.clip(lower=0).rolling(RSI_WINDOW).mean()
    loss  = -delta.clip(upper=0).rolling(RSI_WINDOW).mean()
    df["RSI"] = 100-100/(1+gain/loss.replace(0,np.nan))
    df["ema50"] = df["price"].ewm(span=50).mean()
    df["sma5"]  = df["price"].rolling(5).mean()
    df["sma20"] = df["price"].rolling(20).mean()

    # Simulate grid result next day:
    # If technical filter met → compute drop_pct=ATR/price*100,
    # levels=(10–30 or GRID_PRIMARY for XRP), then
    # entry=df.price[i], next=df.price[i+1], profit if next>entry.
    actions, profits = [], []
    for i in range(len(df)-1):
        atr, adx, rsi, ret = df.loc[i,["ATR","ADX","RSI","return"]]
        tech = (abs(ret)>=atr) and ((adx>=25 and rsi<45) if is_btc else (adx<25 and rsi<30))
        profit = df["price"].iat[i+1] - df["price"].iat[i] if tech else 0
        # Decide label:
        #   RESET  if tech AND profit>0
        #   TERMINATE if tech AND profit<=0
        #   HOLD if not tech
        if tech:
            actions.append(1 if profit>0 else -1)
        else:
            actions.append(0)
        profits.append(profit)
    df = df.iloc[:-1].copy()
    df["action"] = actions        #  1=RESET, 0=HOLD, -1=TERMINATE
    return df.dropna(), profits

btc_labeled, _ = engineer_and_label(btc_df, True)
xrp_labeled, _ = engineer_and_label(xrp_df, False)

# ── Train a single RF to predict {-1,0,1} actions ──
@st.cache_data(ttl=3600)
def train_action_model(df):
    feats = ["return","ATR","ADX","RSI","ema50","sma5","sma20"]
    X, y = df[feats], df["action"]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,shuffle=False)
    m = RandomForestClassifier(n_estimators=200,random_state=0)
    m.fit(Xtr,ytr)
    return m, m.score(Xte,yte)
btc_model, btc_acc = train_action_model(btc_labeled)
xrp_model, xrp_acc = train_action_model(xrp_labeled)

st.sidebar.markdown("### ⚙️ Action Model Accuracy")
st.sidebar.write(f"- BTC-USD: **{btc_acc:.2%}**")
st.sidebar.write(f"- XRP-BTC: **{xrp_acc:.2%}**")

# ── Bootstrap test on simulation of action outcome ──
def bootstrap_action(df, model):
    feats, rates = ["return","ATR","ADX","RSI","ema50","sma5","sma20"], []
    rets = df["return"].values
    for _ in range(100):
        samp = np.random.choice(rets,365,replace=True)
        prices = df["price"].iat[0]*np.exp(np.cumsum(samp/100))
        sim = pd.DataFrame({"price":prices})
        sim["return"]=sim["price"].pct_change()*100
        sim,_ = engineer_and_label(sim, df is btc_labeled)
        wins = (sim["action"]==1).sum()
        total= (sim["action"]!=0).sum()
        rates.append(wins/total if total else np.nan)
    return np.nanmean(rates), np.nanmin(rates), np.nanmax(rates)

btc_mean,btc_min,btc_max = bootstrap_action(btc_df, btc_model)
xrp_mean,xrp_min,xrp_max = bootstrap_action(xrp_df, xrp_model)

st.subheader("🏎️ Bootstraped Action Win‐Rate")
st.write(f"- BTC/USDT reset‐profit %: **{btc_mean:.2%}** (min {btc_min:.2%}, max {btc_max:.2%})")
st.write(f"- XRP/BTC  reset‐profit %: **{xrp_mean:.2%}** (min {xrp_min:.2%}, max {xrp_max:.2%})")

# ── Live action prediction & grid/terminate logic ──
def run_bot(name,pair,price,df_ind,model):
    st.header(f"{name} ({pair})")
    row = df_ind.iloc[-1]
    feat = row[["return","ATR","ADX","RSI","ema50","sma5","sma20"]].values.reshape(1,-1)
    act = int(model.predict(feat)[0])  # -1,0,1
    labels = {1:"🔄 RESET GRID",0:"⏸ HOLD", -1:"🛑 TERMINATE BOT"}
    st.write(f"- **Action:** {labels[act]}")

    if act==1:
        # same grid logic
        atr, ret = row["ATR"], row["return"]
        drop_pct = atr/price*100
        levels = int(np.clip(drop_pct*100,10,30)) if pair=="BTC/USDT" else GRID_PRIMARY
        bot = price*(1-drop_pct/100)
        step= (price-bot)/levels
        per = (usd_alloc/price)/levels
        st.markdown(
            f"**Grid** ({levels})  \n"
            f"- Lower: `{bot:.8f}`  \n"
            f"- Upper: `{price:.8f}`   \n"
            f"- Step: `{step:.8f}`  \n"
            f"- Per: `{per:.6f}` BTC"
        )

btc_live = btc_df["price"].iat[-1]
xrp_live = xrp_df["price"].iat[-1]

run_bot("BTC/USDT Bot","BTC/USDT",btc_live,btc_labeled,btc_model)
run_bot("XRP/BTC Bot",  "XRP/BTC",  xrp_live,xrp_labeled,xrp_model)

# ── About & requirements ──
with st.expander("ℹ️ About"):
    st.markdown("""
    • ML model learned the 3 actions (RESET / HOLD / TERMINATE) from real 365 d history.  
    • Bootstrap-tested 100 simulated years to estimate **profit‐making resets**.  
    • Live, the app now **decides** exactly when to reset, hold, or terminate—no bolt-on.  
    • Copy the grid bounds into Crypto.com Exchange grid bot when “RESET” appears.
    """)

with st.expander("📦 requirements.txt"):
    st.code("""
streamlit
yfinance
pandas
numpy
scikit-learn
pytz
    """)
