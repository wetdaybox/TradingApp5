import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
from datetime import datetime, timedelta

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Initialize session state
if 'manual_price' not in st.session_state:
    st.session_state.manual_price = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now(UK_TIMEZONE).strftime("%H:%M:%S")

@st.cache_data(ttl=300)
def get_realtime_data(pair):
    """Get 48 hours of 5-minute data with proper timezone handling"""
    try:
        data = yf.download(pair, period='2d', interval='5m', progress=False)
        if not data.empty:
            # Convert UTC to London time directly
            data.index = data.index.tz_localize('UTC').tz_convert(UK_TIMEZONE)
            data['RSI'] = get_rsi(data)
            st.session_state.last_update = datetime.now(UK_TIMEZONE).strftime("%H:%M:%S")
        return data
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_fx_rate():
    """Get current FX rate with error handling"""
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='1m')
        return fx_data['Close'].iloc[-1] if not fx_data.empty else 0.80
    except Exception as e:
        st.error(f"FX error: {str(e)}")
        return 0.80

def get_rsi(data, window=14):
    """Robust RSI calculation"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_price_data(pair):
    """Get current price with enhanced validation"""
    data = get_realtime_data(pair)
    fx_rate = get_fx_rate()
    
    if st.session_state.manual_price is not None:
        return st.session_state.manual_price, True
    
    if not data.empty and fx_rate > 0:
        return data['Close'].iloc[-1] / fx_rate, False
    return None, False

def main():
    st.set_page_config(page_title="Crypto Trader Pro", layout="centered")
    st.title("📈 Real-Time Crypto Assistant")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Asset:", CRYPTO_PAIRS)
        use_manual = st.checkbox("Enter Price Manually")
        if use_manual:
            st.session_state.manual_price = st.number_input(
                "Manual Price (£)", min_value=0.01, 
                value=st.session_state.manual_price or 1000.0
            )
        else:
            st.session_state.manual_price = None
            
        account_size = st.number_input("Portfolio Value (£)", 
                                     min_value=100.0, value=1000.0, step=100.0)
        risk_profile = st.select_slider("Risk Profile:", 
                                      options=['Safety First', 'Balanced', 'High Risk'])
        
    with col2:
        st.caption(f"Last update: {st.session_state.last_update} ({UK_TIMEZONE.zone})")
        current_price, is_manual = get_price_data(pair)
        
        if current_price:
            # Rest of the working code from previous version
            # (maintain all the chart and calculation logic)
            # ...
            
if __name__ == "__main__":
    main()
