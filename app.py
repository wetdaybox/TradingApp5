import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Configure the Streamlit app
st.set_page_config(page_title="Crypto Trading Signal Dashboard", layout="wide")

# Sidebar controls
st.sidebar.title("Dashboard Controls")
asset = st.sidebar.selectbox("Select asset", ["BTC-USD", "ETH-USD", "DOGE-USD", "LTC-USD"])
interval = st.sidebar.selectbox("Select timeframe", ["5m", "60m", "1d"], index=2)
rsi_period = st.sidebar.number_input("RSI period", min_value=1, max_value=100, value=14, step=1)
macd_short = st.sidebar.number_input("MACD short EMA", min_value=1, max_value=100, value=12, step=1)
macd_long = st.sidebar.number_input("MACD long EMA", min_value=1, max_value=200, value=26, step=1)
macd_signal = st.sidebar.number_input("MACD signal EMA", min_value=1, max_value=100, value=9, step=1)
sma_window = st.sidebar.number_input("SMA window", min_value=1, max_value=200, value=20, step=1)
toggle_ml = st.sidebar.checkbox("Enable ML Prediction", value=True)
toggle_backtest = st.sidebar.checkbox("Enable Backtest", value=True)

st.title("Crypto Trading Signal Dashboard")

# Fetch historical data
try:
    # Use 60-day window for intraday data (Yahoo limits <1d intervals to 60 days)
    period = "60d" if interval != "1d" else "max"
    data = yf.Ticker(asset).history(period=period, interval=interval)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

if data is None or data.empty:
    st.error("No data fetched. Check the asset or timeframe.")
    st.stop()

# Ensure datetime index
data = data.copy()
data.index = pd.to_datetime(data.index)

# Technical Indicators
def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# RSI
data["RSI"] = calculate_rsi(data["Close"], rsi_period)
# MACD (12/26/9 default but configurable)
short_ema = data["Close"].ewm(span=macd_short, adjust=False).mean()
long_ema  = data["Close"].ewm(span=macd_long, adjust=False).mean()
data["MACD"] = short_ema - long_ema
data["MACD_Signal"] = data["MACD"].ewm(span=macd_signal, adjust=False).mean()
data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]
# SMA
data["SMA"] = data["Close"].rolling(window=sma_window).mean()

# Generate BUY/SELL/HOLD signals
def generate_signal(rsi, macd_hist):
    if pd.isna(rsi) or pd.isna(macd_hist):
        return "HOLD"
    if rsi < 30:
        return "BUY"
    if rsi > 70:
        return "SELL"
    if macd_hist > 0:
        return "BUY"
    if macd_hist < 0:
        return "SELL"
    return "HOLD"

data["Signal"] = [generate_signal(r, m) for r, m in zip(data["RSI"], data["MACD_Hist"])]

# Display the latest signal
current_signal = data["Signal"].iloc[-1]
st.markdown(f"**Current Signal:** {current_signal}")

# Plotly chart: Price (candlestick + SMA), RSI, MACD
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03, row_heights=[0.5, 0.2, 0.3],
                    subplot_titles=("Price + SMA", "RSI", "MACD"))

# Price + SMA
fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], 
                             low=data["Low"], close=data["Close"], name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data["SMA"], mode="lines", name=f"SMA {sma_window}"), row=1, col=1)

# RSI plot with thresholds
fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI"), row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)

# MACD histogram and lines
fig.add_trace(go.Bar(x=data.index, y=data["MACD_Hist"], name="MACD Hist"), row=3, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], mode="lines", name="MACD Line"), row=3, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], mode="lines", name="Signal Line"), row=3, col=1)

fig.update_layout(height=800, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Machine Learning Module (SGDClassifier on RSI & MACD features)
if toggle_ml:
    st.subheader("Machine Learning Prediction")
    df_ml = data.dropna(subset=["RSI", "MACD_Hist"]).copy()
    df_ml["Next_Close"] = df_ml["Close"].shift(-1)
    df_ml = df_ml.dropna(subset=["Next_Close"])
    df_ml["Up"] = (df_ml["Next_Close"] > df_ml["Close"]).astype(int)

    features = df_ml[["RSI", "MACD_Hist"]]
    target = df_ml["Up"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, 
                                                        test_size=0.2, random_state=42)
    model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    st.write(f"Model accuracy (test set): {accuracy:.2%}")

    last_row = data.iloc[-1]
    X_new = np.array([[last_row["RSI"], last_row["MACD_Hist"]]])
    pred = model.predict(X_new)[0]
    st.write(f"Prediction for next interval: **{'Up' if pred == 1 else 'Down'}**")

# Backtesting Module: simulate trades using the signals
if toggle_backtest:
    st.subheader("Strategy Backtest")
    capital = 1000.0
    position = 0
    buy_price = 0.0
    wins = 0
    trades = 0

    for i in range(len(data)):
        sig = data["Signal"].iloc[i]
        price = data["Close"].iloc[i]
        if sig == "BUY" and position == 0:
            position = 1
            buy_price = price
            trades += 1
        elif sig == "SELL" and position == 1:
            position = 0
            profit = price - buy_price
            capital += profit
            if profit > 0:
                wins += 1

    # Close any open position at the end
    if position == 1:
        price = data["Close"].iloc[-1]
        profit = price - buy_price
        capital += profit
        if profit > 0:
            wins += 1

    net_return = (capital - 1000.0) / 1000.0 * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0
    st.write(f"Total return: **{net_return:.2f}%**, Win rate: **{win_rate:.2f}%** ({wins}/{trades} wins)")
