# crypto_dashboard.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
import plotly.graph_objects as go

# --- Streamlit UI ---
st.title("🚀 Crypto Trading Signal Dashboard")

# Sidebar: asset selector and settings
st.sidebar.header("Settings")
asset = st.sidebar.selectbox("Select Asset", ["BTC-USD", "ETH-USD", "DOGE-USD", "SOL-USD"])
rsi_overbought = st.sidebar.slider("RSI Overbought", 50, 100, 70)
rsi_oversold = st.sidebar.slider("RSI Oversold", 0, 50, 30)
enable_pred = st.sidebar.checkbox("Enable Prediction (SGDClassifier)")

# Download price data (cache for performance)
@st.cache_data
def load_data(ticker):
    df = yf.download(tickers=ticker, period="365d", interval="1d")
    return df

df = load_data(asset)
df.dropna(inplace=True)  # drop any NA rows

# --- Compute RSI ---
delta = df['Close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
window = 14
# Wilder’s smoothing: use exponential moving average
avg_gain = up.ewm(alpha=1/window, min_periods=window).mean()
avg_loss = down.ewm(alpha=1/window, min_periods=window).mean()
RS = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + RS))

# --- Compute MACD ---
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['Hist'] = df['MACD'] - df['Signal']

# --- Determine Signals ---
df['SignalText'] = "HOLD"
# Buy: RSI below threshold and MACD rising
df.loc[(df['RSI'] < rsi_oversold) & (df['MACD'] > df['Signal']), 'SignalText'] = "BUY"
# Sell: RSI above threshold and MACD falling
df.loc[(df['RSI'] > rsi_overbought) & (df['MACD'] < df['Signal']), 'SignalText'] = "SELL"

# Get last (most recent) signal
latest_signal = df['SignalText'].iloc[-1]

# --- Optional Prediction ---
pred_text = ""
if enable_pred and len(df) > 30:
    # Prepare features: use RSI and MACD of previous day(s) to predict next return sign
    data = df[['RSI', 'MACD']].dropna().shift(1).dropna()
    data['ReturnNext'] = df['Close'].pct_change().shift(-1)
    data = data.dropna()
    X = data[['RSI','MACD']]
    y = (data['ReturnNext'] > 0).astype(int)  # 1 if next-day up, 0 if down
    model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X, y)
    last_features = np.array(X.tail(1))
    pred = model.predict(last_features)[0]
    prob = model.predict_proba(last_features)[0][int(pred)]
    pred_text = "UP" if pred==1 else "DOWN"
    pred_text += f" ({prob:.0%})"

# --- Display Metrics ---
current_price = df['Close'].iloc[-1]
prev_close = df['Close'].iloc[-2]
delta = current_price - prev_close
st.metric(label=f"{asset} Price", value=f"${current_price:.2f}", delta=f"{delta:.2f}")

st.write(f"**Latest Signal:** {latest_signal}")
if enable_pred:
    st.write(f"**Model Prediction (next day):** {pred_text}")
    # Combine guidance
    if latest_signal == "HOLD":
        guidance = "BUY" if pred_text.startswith("UP") else "SELL"
        st.write(f"**Final Guidance:** {guidance} (technicals neutral)")
    else:
        st.write(f"**Final Guidance:** {latest_signal}")

# --- Plot Candlestick with Signal Annotations ---
fig = go.Figure(data=[go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
)])
# Add buy/sell annotations
for idx, row in df.iterrows():
    if row['SignalText'] == "BUY":
        fig.add_annotation(x=idx, y=row['Low'], text="BUY", 
                           showarrow=True, arrowhead=2, arrowcolor="green", ax=0, ay=-20, bgcolor="white")
    elif row['SignalText'] == "SELL":
        fig.add_annotation(x=idx, y=row['High'], text="SELL", 
                           showarrow=True, arrowhead=2, arrowcolor="red", ax=0, ay=20, bgcolor="white")
fig.update_layout(title=f"{asset} Price with Buy/Sell Signals", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# --- Plot RSI ---
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
# reference lines
fig_rsi.add_hline(y=rsi_overbought, line_color='red', line_dash='dash')
fig_rsi.add_hline(y=rsi_oversold, line_color='green', line_dash='dash')
fig_rsi.update_layout(yaxis_title="RSI", title="RSI Indicator")
st.plotly_chart(fig_rsi, use_container_width=True)

# --- Plot MACD ---
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], mode='lines', name='Signal'))
fig_macd.add_trace(go.Bar(x=df.index, y=df['Hist'], name='Histogram'))
fig_macd.update_layout(yaxis_title="MACD", title="MACD Indicator")
st.plotly_chart(fig_macd, use_container_width=True)
