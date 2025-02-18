import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import backtrader as bt
import requests
import pytz
import time
from datetime import datetime, timedelta
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import plotly.graph_objects as go
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'XRP-USD']
STRATEGIES = ['SMA Crossover', 'RSI Divergence', 'MACD Momentum']
RISK_PARAMS = {
    'max_risk_per_trade': 0.02,
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.10
}
API_COOLDOWN = 65

def generate_sample_data(pair, days=1, interval='5min'):
    """Generate synthetic sample data for initial training"""
    np.random.seed(42)
    base_price = {
        'BTC-USD': 45000,
        'ETH-USD': 2500,
        'XRP-USD': 0.55
    }.get(pair, 100)
    
    dates = pd.date_range(end=datetime.now(), periods=288, freq='5min')
    volatility = 0.02 if pair == 'XRP-USD' else 0.015
    
    prices = base_price * (1 + np.cumsum(volatility * np.random.randn(288)) / 100)
    
    return pd.DataFrame({
        'Open': prices * 0.998,
        'High': prices * 1.005,
        'Low': prices * 0.995,
        'Close': prices,
        'Volume': np.random.randint(1000, 5000, 288)
    }, index=dates)

# Initialize session state with sample data
if 'bot_state' not in st.session_state:
    sample_data = {pair: generate_sample_data(pair) for pair in CRYPTO_PAIRS}
    
    st.session_state.update({
        'bot_state': {
            'positions': {},
            'capital': 10000,
            'historical_data': sample_data,
            'performance': pd.DataFrame(),
            'last_update': datetime.now(pytz.utc),
            'last_api_call': 0,
            'update_in_progress': False,
            'using_sample_data': True  # New flag to track sample data
        },
        'data_cache': {}
    })

# Rest of the code remains the same until the update_market_data function

def safe_update_market_data():
    """Data update that replaces sample data with real data"""
    if st.session_state.bot_state['update_in_progress']:
        return
    
    current_time = time.time()
    time_since_last = current_time - st.session_state.bot_state['last_api_call']
    
    if time_since_last < API_COOLDOWN:
        remaining = API_COOLDOWN - time_since_last
        st.error(f"API cooldown: Please wait {int(remaining)} seconds")
        return
    
    st.session_state.bot_state['update_in_progress'] = True
    
    try:
        with st.spinner("Updating data (replacing sample data)..."):
            new_data = {}
            for pair in CRYPTO_PAIRS:
                data = fetch_market_data(pair)
                if not data.empty:
                    new_data[pair] = data
                time.sleep(API_COOLDOWN)
            
            if new_data:
                st.session_state.bot_state['historical_data'] = new_data
                st.session_state.bot_state['using_sample_data'] = False
                st.success("Real market data loaded! Sample data removed.")
    finally:
        st.session_state.bot_state['update_in_progress'] = False
        st.session_state.bot_state['last_api_call'] = time.time()

# Add sample data disclaimer in the sidebar
with st.sidebar:
    if st.session_state.bot_state['using_sample_data']:
        st.warning("""
        **Initial Sample Data**  
        - Synthetic prices for demonstration  
        - Will be replaced on first update  
        - Not real market data
        """)

# Rest of the code remains the same...
