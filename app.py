import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('trading_bot.log', maxBytes=1e6, backupCount=3)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'USDGBP=X'  # CORRECTED FX PAIR
UK_TIMEZONE = pytz.timezone('Europe/London')
DEFAULT_FX = 0.80
FX_MIN, FX_MAX = 0.70, 0.90  # Realistic USDGBP bounds
MAX_POSITION_SIZE = 1000

@st.cache_data(ttl=60)
def get_realtime_data(pair):
    """Get real-time crypto prices"""
    try:
        data = yf.download(pair, period='1d', interval='1m', progress=False)
        
        # Data freshness check
        if not data.empty:
            last_ts = data.index[-1].to_pydatetime().astimezone(UK_TIMEZONE)
            now = datetime.now(UK_TIMEZONE)
            if (now - last_ts).total_seconds() > 300:
                logger.warning(f"Stale data for {pair}: {last_ts} vs {now}")
                return pd.DataFrame()
        return data
    except Exception as e:
        logger.error(f"Data fetch failed for {pair}", exc_info=True)
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_fx_rate():
    """Get current USD/GBP exchange rate (GBP per 1 USD)"""
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='1m')
        if fx_data.empty:
            return DEFAULT_FX
            
        rate = fx_data['Close'].iloc[-1].item()
        return rate if FX_MIN <= rate <= FX_MAX else DEFAULT_FX
    except Exception as e:
        logger.warning(f"Using default FX rate: {str(e)}")
        return DEFAULT_FX

def get_current_price(pair):
    """Get converted GBP price"""
    data = get_realtime_data(pair)
    fx_rate = get_fx_rate()
    
    if not data.empty:
        usd_price = data['Close'].iloc[-1].item()
        return round(usd_price * fx_rate, 2)  # CORRECTED CONVERSION
    return None

def calculate_levels(pair):
    """Calculate trading levels"""
    data = get_realtime_data(pair)
    if data.empty or len(data) < 20:
        return None
    
    try:
        fx_rate = get_fx_rate()
        closed_data = data.iloc[:-1] if len(data) > 1 else data
        high = closed_data['High'].iloc[-20:].max().item()
        low = closed_data['Low'].iloc[-20:].min().item()
        current_price = data['Close'].iloc[-1].item() * fx_rate  # CORRECTED
        
        stop_loss = max(0.0, low - (high - low) * 0.25) * fx_rate  # CORRECTED
        
        return {
            'buy_zone': round((high + low) / 2 * fx_rate, 2),  # CORRECTED
            'take_profit': round(high + (high - low) * 0.5 * fx_rate, 2),
            'stop_loss': round(stop_loss, 2),
            'current': round(current_price, 2)
        }
    except Exception as e:
        logger.error(f"Level calc failed for {pair}", exc_info=True)
        st.error(f"Calculation error: {str(e)}")
        return None

# ... (rest of the code remains identical to previous version)
