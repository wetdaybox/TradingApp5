import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, timedelta
from pytrends.request import TrendReq
from sklearn.ensemble import RandomForestClassifier
import ta
import time
import requests

# Configuration with multiple fallback sources
CRYPTO_PAIRS = {
    'BTC-USD': {'coingecko': 'bitcoin', 'ccxt': 'BTC/USD'},
    'ETH-USD': {'coingecko': 'ethereum', 'ccxt': 'ETH/USD'},
    'BNB-USD': {'coingecko': 'binancecoin', 'ccxt': 'BNB/USD'},
    'XRP-USD': {'coingecko': 'ripple', 'ccxt': 'XRP/USD'},
    'ADA-USD': {'coingecko': 'cardano', 'ccxt': 'ADA/USD'}
}
EXCHANGE_RATE_SOURCES = [
    {'name': 'Yahoo', 'ticker': 'GBPUSD=X'},
    {'name': 'ECB', 'url': 'https://api.exchangerate.host/latest?base=USD'},
    {'name': 'Backup', 'rate': 0.79}
]

# Initialize session state with data persistence
session_defaults = {
    'trades': [],
    'model': None,
    'gbp_rate': None,
    'cached_data': {},
    'data_source': 'Yahoo'
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def robust_data_fetch(ticker, period='3d', interval='30m'):
    """Fetch data with multiple fallback strategies and caching"""
    cache_key = f"{ticker}_{period}_{interval}"
    
    # Return cached data if available
    if cache_key in st.session_state.cached_data:
        return st.session_state.cached_data[cache_key]
    
    max_retries = 3
    sources = [
        ('Yahoo', lambda: yf.download(ticker, period=period, interval=interval)),
        ('CoinGecko', lambda: fetch_coingecko_data(ticker, period)),
        ('CCXT', lambda: fetch_ccxt_data(ticker, period, interval))
    ]
    
    for source_name, fetch_func in sources:
        for attempt in range(max_retries):
            try:
                data = fetch_func()
                if validate_data(data):
                    st.session_state.data_source = source_name
                    data = process_data(data, interval)
                    st.session_state.cached_data[cache_key] = data
                    return data
            except Exception as e:
                st.warning(f"{source_name} attempt {attempt+1} failed: {str(e)}")
                time.sleep(2 ** attempt)
    
    st.error("All data sources failed. Using last known good data.")
    return pd.DataFrame()

def fetch_coingecko_data(ticker, period):
    """Fetch data from CoinGecko API with OHLC format"""
    crypto_id = CRYPTO_PAIRS[ticker]['coingecko']
    days = {'3d': 3, '5d': 5, '30d': 30}[period]
    
    # Get OHLC data
    ohlc_url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/ohlc?vs_currency=usd&days={days}"
    response = requests.get(ohlc_url, timeout=10)
    response.raise_for_status()
    
    # Convert to DataFrame
    ohlc_data = response.json()
    df = pd.DataFrame(ohlc_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    return df[['Open', 'High', 'Low', 'Close']]

def fetch_ccxt_data(ticker, period, interval):
    """Fetch data from CCXT (Binance)"""
    # Implement CCXT integration here
    raise NotImplementedError("CCXT integration pending")

def validate_data(data):
    """Ensure data meets minimum quality standards"""
    if data.empty:
        return False
    required_cols = ['Open', 'High', 'Low', 'Close']
    return all(col in data.columns for col in required_cols)

def get_exchange_rate():
    """Get GBP/USD rate with multiple fallbacks"""
    for source in EXCHANGE_RATE_SOURCES:
        try:
            if source['name'] == 'Yahoo':
                rate = yf.download(source['ticker'], period='1d')['Close'].iloc[-1]
            elif source['name'] == 'ECB':
                response = requests.get(source['url'], timeout=5)
                rate = response.json()['rates']['GBP']
            else:
                rate = source['rate']
            
            if 0.5 < rate < 2.0:
                return rate
        except:
            continue
    return 0.79  # Final fallback

# Rest of the functions (convert_to_gbp, train_model, advanced_analysis) 
# remain similar but use the new data fetching system

def main():
    st.set_page_config(page_title="Pro Crypto Trader", layout="wide")
    
    # Status dashboard
    with st.sidebar:
        st.subheader("System Status")
        st.write(f"Data Source: {st.session_state.data_source}")
        st.write(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"Exchange Rate: £1 = ${get_exchange_rate():.2f}")
    
    # Main interface
    col1, col2 = st.columns([1, 3])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", list(CRYPTO_PAIRS.keys()))
        # ... rest of UI elements
    
    with col2:
        analysis = advanced_analysis(pair)
        # ... display analysis

if __name__ == "__main__":
    main()
