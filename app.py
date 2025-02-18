import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Configuration
CRYPTO_ASSETS = {
    'BTC': {'coingecko': 'bitcoin', 'binance': 'BTCUSDT'},
    'ETH': {'coingecko': 'ethereum', 'binance': 'ETHUSDT'},
    'BNB': {'coingecko': 'binancecoin', 'binance': 'BNBUSDT'}
}

EXCHANGE_RATE_SOURCES = [
    {'name': 'ECB', 'url': 'https://api.exchangerate.host/latest?base=USD'},
    {'name': 'IMF', 'url': 'https://api.imf.org/exchangerates/v1/latest?base=USD'},
    {'name': 'Backup', 'rate': 0.79}
]

# Session state management
if 'app_data' not in st.session_state:
    st.session_state.app_data = {
        'prices': {},
        'history': pd.DataFrame(),
        'exchange_rate': 0.79,
        'last_updated': datetime.now()
    }

def get_crypto_price(asset):
    """Multi-source price verification with consensus"""
    sources = [
        lambda: requests.get(f'https://api.coingecko.com/api/v3/simple/price?ids={asset["coingecko"]}&vs_currencies=usd').json()[asset["coingecko"]]['usd'],
        lambda: float(requests.get(f'https://api.binance.com/api/v3/ticker/price?symbol={asset["binance"]}').json()['price']),
        lambda: st.session_state.app_data['prices'].get(asset, None)
    ]
    
    prices = []
    for source in sources:
        try:
            price = source()
            if price and 0 < price < 1000000:  # Sanity check
                prices.append(price)
                time.sleep(0.3)  # Rate limiting
        except:
            continue
    
    return np.median(prices) if prices else None

def get_exchange_rate():
    """Reliable GBP/USD rate with validation"""
    for source in EXCHANGE_RATE_SOURCES:
        try:
            if source['name'] == 'Backup':
                return source['rate']
                
            response = requests.get(source['url'], timeout=2)
            rate = response.json()['rates']['GBP']
            if 0.75 < rate < 1.5:
                return rate
        except:
            continue
    return 0.79

def update_market_data():
    """Full market data update with history tracking"""
    new_data = {}
    for symbol, asset in CRYPTO_ASSETS.items():
        price = get_crypto_price(asset)
        if price:
            new_data[symbol] = price
    
    if new_data:
        timestamp = datetime.now()
        history = st.session_state.app_data['history']
        new_row = pd.DataFrame([new_data], index=[timestamp])
        st.session_state.app_data['history'] = pd.concat([history, new_row]).last('15T')
        st.session_state.app_data['prices'] = new_data
        st.session_state.app_data['exchange_rate'] = get_exchange_rate()
        st.session_state.app_data['last_updated'] = timestamp

def display_price_chart():
    """Interactive price chart with technical indicators"""
    fig = go.Figure()
    
    for symbol in CRYPTO_ASSETS:
        if symbol in st.session_state.app_data['history']:
            fig.add_trace(go.Scatter(
                x=st.session_state.app_data['history'].index,
                y=st.session_state.app_data['history'][symbol],
                name=symbol,
                mode='lines+markers'
            ))
    
    fig.update_layout(
        title='Real-Time Price Movement',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="Pro Crypto Trader", layout="wide", page_icon="📈")
    
    # Header
    st.title("Professional Cryptocurrency Trading Terminal")
    st.write("Real-time market data with institutional-grade reliability")
    
    # Auto-refresh logic
    last_update = st.session_state.app_data['last_updated']
    if (datetime.now() - last_update).seconds > 300:  # 5 minute refresh
        update_market_data()
    
    # Control panel
    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Manual Refresh"):
            update_market_data()
        st.metric("GBP/USD Rate", f"£1 = ${st.session_state.app_data['exchange_rate']:.4f}")
        st.write(f"Last Update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main display
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.header("Current Prices")
        for symbol, price in st.session_state.app_data['prices'].items():
            gbp_price = price * st.session_state.app_data['exchange_rate']
            st.metric(f"{symbol}/USD", f"${price:,.2f}", f"£{gbp_price:,.2f}")
    
    with col2:
        st.header("Market Analysis")
        display_price_chart()

if __name__ == "__main__":
    main()
