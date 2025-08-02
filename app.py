import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import pytz

# Constants
ATR_WINDOW = 14
ADX_WINDOW = 14
RSI_WINDOW = 14
MIN_VOL = 2.0

# Fetch data
def load_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d")[["Close"]].rename(columns={"Close": "price"})
    df["return"] = df["price"].pct_change() * 100
    return df.dropna()

btc_df = load_data("BTC-USD")
xrp_df = load_data("XRP-USD")

# Fix for engineer_and_label
def engineer_and_label(df_raw: pd.DataFrame, is_btc: bool):
    df = df_raw.copy().reset_index(drop=True)

    df["ATR"] = df["return"].abs().rolling(ATR_WINDOW).mean()
    up = df["price"].diff().clip(lower=0).rolling(ADX_WINDOW).mean()
    down = -df["price"].diff().clip(upper=0).rolling(ADX_WINDOW).mean()
    df["ADX"] = (abs(up - down) / (up + down)).rolling(ADX_WINDOW).mean() * 100
    delta = df["price"].diff()
    gain = delta.clip(lower=0).rolling(RSI_WINDOW).mean()
    loss = -delta.clip(upper=0).rolling(RSI_WINDOW).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df["ema50"] = df["price"].ewm(span=50).mean()
    df["sma5"] = df["price"].rolling(5).mean()
    df["sma20"] = df["price"].rolling(20).mean()

    actions = []
    for i in range(len(df) - 1):
        atr = df["ATR"].iloc[i]
        adx = df["ADX"].iloc[i]
        rsi = df["RSI"].iloc[i]
        ret = df["return"].iloc[i]

        if is_btc:
            tech = (abs(ret) >= atr) and (adx >= 25) and (rsi < 45)
        else:
            tech = (abs(ret) >= atr) and (adx < 25) and (rsi < 30)

        if not tech:
            actions.append(0)
            continue

        entry = df["price"].iloc[i]
        nxt = df["price"].iloc[i + 1]
        profit = nxt - entry

        actions.append(1 if profit > 0 else -1)

    df = df.iloc[:-1].copy()
    df["action"] = actions
    return df.dropna()

btc_labeled = engineer_and_label(btc_df, True)
xrp_labeled = engineer_and_label(xrp_df, False)

# Combine and train
all_data = pd.concat([btc_labeled, xrp_labeled], ignore_index=True)
features = ["ATR", "ADX", "RSI", "return", "ema50", "sma5", "sma20"]
X = all_data[features]
y = all_data["action"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit display
st.set_page_config(layout="centered")
st.title("📈 Grid Trading Bot ML Strategy Tester")
now = datetime.now(pytz.timezone("Europe/London"))
st.caption(f"Last updated: {now:%Y-%m-%d %H:%M:%S %Z}")

st.subheader("Model Prediction Accuracy on Training Data")
st.write(f"Accuracy: {model.score(X, y)*100:.2f}%")

# Predict latest action
latest = all_data.iloc[-1:][features]
pred = model.predict(latest)[0]
labels = {-1: "TERMINATE", 0: "HOLD", 1: "RESET"}

st.subheader("📊 Latest Model Recommendation")
st.metric("Recommended Action", labels[pred])
