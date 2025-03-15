import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime

# Configuration
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'BNB-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')

@st.cache_data(ttl=300)
def get_realtime_price(pair):
    """Get real-time crypto prices in GBP"""
    try:
        data = yf.Ticker(pair).history(period='1d', interval='1m')
        if not data.empty:
            # Proper conversion from pandas/numpy types
            return float(data['Close'].iloc[-1].item())  
        return None
    except Exception as e:
        st.error(f"Price error: {str(e)}")
        return None

@st.cache_data(ttl=300)
def download_data(pair, period='1d', interval='15m'):
    """Download historical data"""
    try:
        return yf.download(pair, period=period, interval=interval, progress=False)
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def calculate_levels(pair):
    """Calculate trading levels with proper type handling"""
    data = download_data(pair)
    if data.empty or len(data) < 20:
        return None
    
    try:
        closed_data = data.iloc[:-1] if len(data) > 1 else data
        
        # Proper Series handling with .item()
        high = closed_data['High'].iloc[-20:].max().item()
        low = closed_data['Low'].iloc[-20:].min().item()
        current_price = data['Close'].iloc[-1].item()

        stop_loss = max(0.0, low - (high - low) * 0.25)
        
        return {
            'buy_zone': round((high + low)/2, 2),
            'take_profit': round(high + (high-low)*0.5, 2),
            'stop_loss': round(stop_loss, 2),
            'current': current_price
        }
    except Exception as e:
        st.error(f"Level calculation error: {str(e)}")
        return None

# Rest of the functions remain unchanged...

if __name__ == "__main__":
    main()
