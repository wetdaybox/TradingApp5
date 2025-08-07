# Final Crypto.com-Compatible Grid Bot Assistant
# Streamlit app designed to guide manual setup of Grid Bot in the UK.
# Removed volatility-adaptive grid spacing for compliance with platform.

import streamlit as st
import pandas as pd, numpy as np
import requests, time
from datetime import datetime
import pytz
import altair as alt
from sklearn.linear_model import SGDClassifier

# Settings
H_DAYS = 90
CLASS_THRESH = 0.80

# Fetch price history
def fetch_history(symbol):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {"vs_currency": "usd", "days": H_DAYS}
        res = requests.get(url, params=params, timeout=10)
        data = res.json()["prices"]
        df = pd.DataFrame(data, columns=["ts", "price"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("date").resample("D").last()
        df["return"] = df["price"].pct_change() * 100
        return df.dropna()
    except:
        return pd.DataFrame()

# Indicators
def compute_ind(df):
    df["ema"] = df["price"].ewm(span=20).mean()
    df["sma"] = df["price"].rolling(20).mean()
    df["vol"] = df["return"].rolling(14).std().fillna(0)
    return df.dropna()

# Fetch current price
def fetch_live(symbol):
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price", params={
            "ids": symbol, "vs_currencies": "usd"
        })
        return r.json()[symbol]["usd"]
    except:
        return np.nan

# ML Feature
def today_feat(df):
    i = -1
    return [[
        df["return"].iloc[i],
        df["vol"].iloc[i],
        df["price"].iloc[i] - df["ema"].iloc[i],
        df["price"].iloc[i] - df["sma"].iloc[i],
        df["price"].iloc[i]
    ]]

# ───── UI Setup ─────
st.set_page_config(layout="wide")
st.title("🇬🇧 Grid Bot Assistant for Crypto.com (UK Compatible)")
st.caption(f"Last updated: {datetime.now(pytz.timezone('Europe/London')).strftime('%Y-%m-%d %H:%M %Z')}")

# Sidebar Inputs
st.sidebar.header("🛠️ Strategy Inputs")
invest = st.sidebar.number_input("Total Investment ($)", 100.0, 10000.0, 3000.0, 100.0)
grid_count = st.sidebar.slider("Grid Levels", 2, 30, 10)
spacing_pct = st.sidebar.slider("Grid Spacing (%)", 0.1, 10.0, 2.0, 0.1)
stop_pct = st.sidebar.slider("Stop-Loss (%)", 1.0, 10.0, 3.0)
take_pct = st.sidebar.slider("Take-Profit (%)", 1.0, 20.0, 6.0)

# Load Data
df = compute_ind(fetch_history("bitcoin"))
live_price = fetch_live("bitcoin")

if df.empty or np.isnan(live_price):
    st.error("⚠️ Failed to fetch price data.")
    st.stop()

# Grid Calculation
grid_range = spacing_pct * grid_count / 100
lower = live_price * (1 - grid_range)
upper = live_price * (1 + grid_range)
grid_prices = np.linspace(lower, upper, grid_count)

# ML Model
if "clf" not in st.session_state:
    model = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
    model.partial_fit(np.zeros((2,5)), [0,1], classes=[0,1])
    st.session_state.clf = model

prob = st.session_state.clf.predict_proba(today_feat(df))[0][1]
st.metric("🤖 ML Buy Probability", f"{prob:.2%}")

# Chart
st.subheader("📊 BTC Price with Grid Levels")
df_plot = df.reset_index()
base = alt.Chart(df_plot).mark_line().encode(x="date:T", y="price:Q")
rules = alt.Chart(pd.DataFrame({"y": grid_prices})).mark_rule(
    color="gray", strokeDash=[4,2]).encode(y="y:Q")
st.altair_chart(base + rules, use_container_width=True)

# Backtest
st.subheader("🔁 Backtest (Simulation Over 90 Days)")
bal = invest
wins, losses, trades = 0, 0, 0
in_pos = False
entry = 0.0

for p in df["price"]:
    if not in_pos and p <= lower:
        in_pos = True
        entry = p
        trades += 1
    elif in_pos:
        if p >= entry * (1 + take_pct/100):
            profit = (p - entry) / entry * bal
            bal += profit
            wins += 1
            in_pos = False
        elif p <= entry * (1 - stop_pct/100):
            loss = (entry - p) / entry * bal
            bal -= loss
            losses += 1
            in_pos = False

pnl = bal - invest
st.write(f"**Total Trades:** {trades}")
st.write(f"✅ Wins: {wins} | ❌ Losses: {losses}")
st.write(f"📈 Net PnL: ${pnl:.2f} ({(pnl/invest):.2%})")

st.markdown("---")
st.info("This assistant is for strategy planning only and matches settings allowed by the Crypto.com Grid Bot in the UK.")
