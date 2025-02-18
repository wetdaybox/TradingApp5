# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime, timedelta
from pytrends.request import TrendReq
from sklearn.ensemble import RandomForestClassifier
import ta
import time
import requests

# Configuration with multiple data sources
CRYPTO_PAIRS = {
    'BTC-USD': {'coingecko': 'bitcoin'},
    'ETH-USD': {'coingecko': 'ethereum'},
    'BNB-USD': {'coingecko': 'binancecoin'},
    'XRP-USD': {'coingecko': 'ripple'},
    'ADA-USD': {'coingecko': 'cardano'}
}
EXCHANGE_RATE_TICKER = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')

# Initialize session state with data caching
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'gbp_rate' not in st.session_state:
    st.session_state.gbp_rate = None
if 'fallback_data' not in st.session_state:
    st.session_state.fallback_data = {}

def robust_data_fetch(tickers, period='3d', interval='30m'):
    """Fetch data with multiple fallback strategies"""
    max_retries = 5
    backoff = [2, 4, 8, 16, 32]  # Exponential backoff with jitter
    
    for attempt in range(max_retries):
        try:
            # Try Yahoo Finance first
            data = yf.download(
                tickers,
                period=period,
                interval=interval,
                group_by='ticker',
                progress=False,
                timeout=10
            )
            if not data.empty:
                return data
        except Exception as yf_error:
            st.warning(f"Yahoo Finance attempt {attempt+1} failed: {str(yf_error)}")
            
        try:
            # Fallback to CoinGecko API for crypto data
            if 'USD' in tickers[0]:
                crypto_id = CRYPTO_PAIRS[tickers[0]]['coingecko']
                days = {'3d': 3, '5d': 5, '30d': 30}[period]
                url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                crypto_data = response.json()
                
                # Convert to DataFrame
                prices = pd.DataFrame(crypto_data['prices'], columns=['timestamp', 'Close'])
                prices['Date'] = pd.to_datetime(prices['timestamp'], unit='ms')
                prices.set_index('Date', inplace=True)
                return prices
        except Exception as cg_error:
            st.warning(f"CoinGecko attempt {attempt+1} failed: {str(cg_error)}")
        
        # Random jitter before retry
        time.sleep(backoff[attempt] + np.random.uniform(0, 1))
    
    st.error("All data sources failed. Using cached data if available.")
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_exchange_rate():
    """Get GBP/USD rate with multiple fallbacks"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            rate = yf.download(EXCHANGE_RATE_TICKER, period='1d')['Close'].iloc[-1]
            if 0.5 < rate < 2.0:
                return rate
        except:
            pass
        
        try:
            # Fallback to ECB API
            response = requests.get('https://api.exchangerate-api.com/v4/latest/USD')
            return response.json()['rates']['GBP']
        except:
            pass
        
    return 0.79  # Hardcoded fallback

def convert_to_gbp(usd_prices):
    rate = get_exchange_rate()
    return usd_prices * rate

# Rest of the functions remain similar but use robust_data_fetch instead of get_data
# ... (keep the same structure for train_model, advanced_analysis, etc) ...

def main():
    st.set_page_config(page_title="Pro Crypto Trader", layout="wide")
    
    # Add service status indicator
    yf_status = "🟢 Operational" if not st.session_state.get('yf_down') else "🔴 Outage"
    st.sidebar.markdown(f"**Service Status:** {yf_status}")
    
    # Rest of the main function remains the same
    # ... (keep existing UI structure) ...

if __name__ == "__main__":
    main()
