import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
import requests
from datetime import datetime
from typing import Dict, Optional

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
UK_TIMEZONE = pytz.timezone('Europe/London')

# Configure requests session with proper headers
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
})

@st.cache_data(ttl=300)
def download_data(pair: str, period: str = '1d', interval: str = '15m') -> pd.DataFrame:
    """Robust data download with error handling"""
    try:
        # Get GBP/USD exchange rate
        fx_data = yf.download('GBPUSD=X', period='1d', interval='1m', session=session)
        fx_rate = fx_data['Close'].iloc[-1] if not fx_data.empty else 0.80
        
        # Download crypto data
        data = yf.download(
            tickers=pair,
            period=period,
            interval=interval,
            session=session,
            progress=False
        )
        
        if not data.empty:
            # Convert USD to GBP
            data[['Open', 'High', 'Low', 'Close']] *= fx_rate
            return data.round(2)
        return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def calculate_levels(pair: str) -> Optional[Dict[str, float]]:
    """Calculate trading levels with validation"""
    data = download_data(pair)
    if data.empty or len(data) < 20:
        return None
    
    try:
        closed_data = data.iloc[:-1] if len(data) > 1 else data
        high = closed_data['High'].iloc[-20:].max()
        low = closed_data['Low'].iloc[-20:].min()
        current = data['Close'].iloc[-1]
        
        if any(np.isnan([high, low, current])):
            return None
            
        return {
            'buy_zone': round((high + low)/2, 2),
            'take_profit': round(high + (high-low)*0.5, 2),
            'stop_loss': round(max(0, low - (high-low)*0.25), 2),
            'current': round(current, 2)
        }
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return None

# Rest of the functions (calculate_position_size, calculate_technical_indicators, main) 
# remain identical to the previous working version...

if __name__ == "__main__":
    main()
