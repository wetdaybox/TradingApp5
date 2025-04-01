import warnings
import traceback
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pytz
import joblib
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore", category=FutureWarning)

# ======================================================
# Configuration
# ======================================================
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
REFRESH_INTERVAL = 60  # seconds
MODEL_PATH = "sgd_classifier.pkl"

# ======================================================
# Core Functions
# ======================================================
def get_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100/(1+rs))

def get_realtime_data(pair):
    try:
        data = yf.download(pair, period='7d', interval='5m', progress=False)
        if not data.empty:
            if 'Adj Close' in data.columns:
                data.rename(columns={'Adj Close': 'Close'}, inplace=True)
            
            data['RSI'] = get_rsi(data)
            data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
            data['SMA'] = data['Close'].rolling(20).mean()
            
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            data.index = data.index.tz_convert(UK_TIMEZONE)
            
            return data
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Data error: {e}")
        return pd.DataFrame()

# ======================================================
# Display Components
# ======================================================
def display_price(current_price, is_manual):
    try:
        price_str = f"£{current_price:,.4f}" if current_price > 1 else f"£{current_price:.8f}"
        st.metric(
            label="Current Price (" + ("Manual" if is_manual else "Live") + ")",
            value=price_str,
            delta="Manual override" if is_manual else "Real-time data"
        )
    except Exception as e:
        st.error(f"Price display error: {e}")

def display_chart(data):
    try:
        if not data.empty and all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                increasing_line_color='#2e7bcf',
                decreasing_line_color='#cf2e2e'
            )])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chart data loading...")
    except Exception as e:
        st.error(f"Chart error: {e}")

# ======================================================
# Main Application
# ======================================================
def main():
    st.title("🚀 Revolutionary Crypto Trading Bot")
    st.markdown("**Free-to-use, advanced crypto trading assistant**")
    
    try:
        if 'manual_price' not in st.session_state:
            st.session_state.manual_price = None
            
        pair = st.sidebar.selectbox("Select Asset:", CRYPTO_PAIRS)
        use_manual = st.sidebar.checkbox("Manual Price Override")
        
        if use_manual:
            st.session_state.manual_price = st.sidebar.number_input(
                "Enter Price (£)", min_value=0.0, value=1000.0)
        
        if st.session_state.manual_price is not None:
            current_price = float(st.session_state.manual_price)
            is_manual = True
        else:
            data = get_realtime_data(pair)
            fx_data = yf.download(FX_PAIR, period='1d', progress=False)
            fx_rate = fx_data['Close'].iloc[-1] if not fx_data.empty else 0.80
            current_price = data['Close'].iloc[-1] / fx_rate if not data.empty else None
            is_manual = False
        
        if current_price is not None:
            display_price(current_price, is_manual)
            if not is_manual:
                display_chart(data)
        else:
            st.warning("Waiting for market data...")
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    st.set_page_config(page_title="Crypto Trading Bot", layout="wide")
    st_autorefresh(interval=REFRESH_INTERVAL*1000, key="data_refresh")
    main()
