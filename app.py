import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
from datetime import datetime

# Configuration
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'BNB-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Initialize session state
if 'manual_price' not in st.session_state:
    st.session_state.manual_price = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now(UK_TIMEZONE).strftime("%H:%M:%S")

@st.cache_data(ttl=30)
def get_realtime_data(pair):
    """Get 48 hours of 5-minute data"""
    try:
        data = yf.download(pair, period='2d', interval='5m', progress=False, auto_adjust=True)
        if not data.empty:
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC').tz_convert(UK_TIMEZONE)
            else:
                data.index = data.index.tz_convert(UK_TIMEZONE)
            data['RSI'] = get_rsi(data)
            st.session_state.last_update = datetime.now(UK_TIMEZONE).strftime("%H:%M:%S")
        return data
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def get_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_price_data(pair):
    data = get_realtime_data(pair)
    
    if st.session_state.manual_price is not None:
        return float(st.session_state.manual_price), True
    
    if not data.empty:
        # Convert to native Python float
        return data['Close'].iloc[-1].item(), False  # Changed to .item()
    return None, False

def calculate_levels(pair, current_price):
    """24-hour range calculation"""
    data = get_realtime_data(pair)
    if data.empty or len(data) < 288:
        return None
    
    try:
        full_day_data = data.iloc[-288:]
        recent_low = full_day_data['Low'].min().item()  # Convert to scalar
        recent_high = full_day_data['High'].max().item()  # Convert to scalar
        last_rsi = data['RSI'].iloc[-1].item()  # Convert to scalar
        
        return {
            'buy_zone': round(recent_low * 0.98, 2),
            'take_profit': round(current_price * 1.15, 2),
            'stop_loss': round(current_price * 0.95, 2),
            'rsi': round(last_rsi, 1),
            'high': round(recent_high, 2),
            'low': round(recent_low, 2)
        }
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return None

# Rest of the code remains identical to previous working version
# ... [Keep all other functions and main() implementation the same]

if __name__ == "__main__":
    main()
