import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime

# Configuration (CHANGED MAX AGE TO 10)
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
FX_MIN = 1.20
FX_MAX = 1.40
DEFAULT_FX = 1.25
MAX_DATA_AGE_MIN = 10  # Changed from 5 to 10

@st.cache_data(ttl=60)
def get_realtime_data(pair):
    try:
        data = yf.download(pair, period='1d', interval='1m', progress=False)
        
        if not data.empty:
            last_ts = data.index[-1].to_pydatetime().replace(tzinfo=pytz.UTC)
            now = datetime.now(pytz.UTC)
            age_min = (now - last_ts).total_seconds() / 60
            
            if age_min > MAX_DATA_AGE_MIN:
                st.warning(f"Data is {age_min:.1f} minutes old - using historical data")  # Changed message
                return data  # Return data anyway instead of empty DF
        
        return data
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

# REST OF THE CODE REMAINS IDENTICAL TO PREVIOUS WORKING VERSION
# (Only changed MAX_DATA_AGE_MIN and get_realtime_data() warning)
