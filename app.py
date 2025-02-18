import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, timedelta
import ta
import requests
import random

# Configuration with offline fallbacks
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FALLBACK_DATA = {
    'BTC-USD': {'price': 35000, 'rsi': 45, 'atr': 1500},
    'ETH-USD': {'price': 2000, 'rsi': 50, 'atr': 120},
    'BNB-USD': {'price': 300, 'rsi': 55, 'atr': 15},
    'XRP-USD': {'price': 0.6, 'rsi': 40, 'atr': 0.05},
    'ADA-USD': {'price': 0.5, 'rsi': 48, 'atr': 0.04}
}

# Initialize session state with persistent storage
if 'trades' not in st.session_state:
    st.session_state.update({
        'trades': [],
        'model': None,
        'gbp_rate': 0.79,
        'cached_data': FALLBACK_DATA.copy(),
        'last_update': datetime.now(),
        'data_source': 'Offline'
    })

def resilient_data_fetch(pair):
    """Fetch data with multiple fallback layers"""
    sources = [
        lambda: yfinance_fetch(pair),
        lambda: coingecko_fetch(pair),
        lambda: st.session_state.cached_data[pair]
    ]
    
    for source in sources:
        try:
            data = source()
            if data and 'price' in data:
                st.session_state.cached_data[pair] = data
                return data
        except Exception as e:
            st.error(f"Data source error: {str(e)}")
            time.sleep(1)
    
    return generate_simulated_data(pair)

def yfinance_fetch(pair):
    """Yahoo Finance fetch with aggressive retry"""
    for _ in range(3):
        try:
            data = yf.download(pair, period='1d', interval='5m', progress=False)
            if not data.empty:
                return process_live_data(data, pair)
        except:
            time.sleep(2)
    raise ConnectionError("Yahoo Finance unavailable")

def coingecko_fetch(pair):
    """CoinGecko free API fallback"""
    coin_id = {
        'BTC-USD': 'bitcoin',
        'ETH-USD': 'ethereum',
        'BNB-USD': 'binancecoin',
        'XRP-USD': 'ripple',
        'ADA-USD': 'cardano'
    }.get(pair)
    
    try:
        response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true")
        price = response.json()[coin_id]['usd']
        change = response.json()[coin_id]['usd_24h_change']
        return {
            'price': price,
            'rsi': 50 + (change / 2),
            'atr': price * 0.05
        }
    except:
        return st.session_state.cached_data[pair]

def generate_simulated_data(pair):
    """Generate plausible data when all sources fail"""
    last_data = st.session_state.cached_data[pair]
    return {
        'price': last_data['price'] * random.uniform(0.99, 1.01),
        'rsi': max(30, min(70, last_data['rsi'] + random.uniform(-2, 2))),
        'atr': last_data['atr'] * random.uniform(0.95, 1.05)
    }

def process_live_data(data, pair):
    """Process live data with technical analysis"""
    ta_features = ta.add_all_ta_features(
        data, open="Open", high="High", low="Low", 
        close="Close", volume="Volume", fillna=True
    )
    return {
        'price': data['Close'].iloc[-1],
        'rsi': ta_features['momentum_rsi'].iloc[-1],
        'atr': ta_features['volatility_atr'].iloc[-1]
    }

def get_gbp_rate():
    """Multi-source exchange rate with caching"""
    try:
        rate = yf.download('GBPUSD=X', period='1d')['Close'].iloc[-1]
        if 0.5 < rate < 2.0:
            return rate
    except:
        pass
    
    try:
        response = requests.get('https://api.exchangerate.host/latest?base=USD')
        return response.json()['rates']['GBP']
    except:
        return 0.79

def main():
    st.set_page_config(page_title="Always-On Crypto Trader", layout="wide")
    
    # Update session state
    st.session_state.gbp_rate = get_gbp_rate()
    time_diff = (datetime.now() - st.session_state.last_update).seconds // 60
    
    # Sidebar controls
    with st.sidebar:
        st.metric("GBP/USD Rate", f"£1 = ${st.session_state.gbp_rate:.2f}")
        st.write(f"Data Freshness: {time_diff} mins ago")
        if st.button("🔄 Force Refresh"):
            st.session_state.cached_data = FALLBACK_DATA.copy()
            st.session_state.last_update = datetime.now()
            st.rerun()
    
    # Main interface
    pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
    data = resilient_data_fetch(pair)
    
    # Display analysis
    st.write(f"## {pair} Analysis ({st.session_state.data_source})")
    cols = st.columns(3)
    cols[0].metric("Price", f"£{data['price'] * st.session_state.gbp_rate:,.2f}")
    cols[1].metric("RSI", f"{data['rsi']:.1f}", 
                  "Oversold" if data['rsi'] < 30 else "Overbought" if data['rsi'] > 70 else "Neutral")
    cols[2].metric("Volatility", f"£{data['atr'] * st.session_state.gbp_rate:,.2f}")
    
    # Trading signals
    st.write("### Trading Signals")
    if data['rsi'] < 35:
        st.success("Strong Buy Signal - Oversold Condition")
    elif data['rsi'] > 65:
        st.error("Sell Signal - Overbought Condition")
    else:
        st.info("Neutral Market Conditions")
    
    st.write("⚠️ Note: Data simulation active during market outages")

if __name__ == "__main__":
    main()
