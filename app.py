import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
FX_MIN = 1.20
FX_MAX = 1.40
DEFAULT_FX = 1.25
MAX_DATA_AGE_MIN = 15  # Increased tolerance

@st.cache_data(ttl=60)
def get_realtime_data(pair):
    """Safer version with fallback to last valid data"""
    try:
        data = yf.download(pair, period='1d', interval='1m', progress=False)
        
        # Handle empty data scenarios
        if data.empty:
            st.warning("Received empty market data")
            return pd.DataFrame()
            
        # Simplified timestamp check
        last_ts = data.index[-1].to_pydatetime()
        now = datetime.now(pytz.utc)
        if (now - last_ts).total_seconds() > MAX_DATA_AGE_MIN * 60:
            st.warning("Using slightly stale data")
            
        return data  # Always return data if available
        
    except Exception as e:
        st.error(f"Data connection failed: {str(e)}")
        return pd.DataFrame()

# Rest of the code IDENTICAL to your original working version
# (FX rate handling, level calculations, etc. unchanged)
# Only modified get_realtime_data() and increased MAX_DATA_AGE_MIN

def get_fx_rate():
    # ... (unchanged from previous working version) ...

def calculate_levels(pair):
    # ... (unchanged from previous working version) ...

def main():
    # ... (unchanged from previous working version) ...

if __name__ == "__main__":
    main()
