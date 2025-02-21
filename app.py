import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from bs4 import BeautifulSoup
import requests
import time
import feedparser
import os

# Configuration
CRYPTO_PAIRS = {
    'BTC-GBP': {'slug': 'bitcoin', 'selector': '.priceValue span'},
    'ETH-GBP': {'slug': 'ethereum', 'selector': '.priceValue span'},
    'BNB-GBP': {'slug': 'bnb', 'selector': '.priceValue span'},
    'XRP-GBP': {'slug': 'ripple', 'selector': '.priceValue span'},
    'ADA-GBP': {'slug': 'cardano', 'selector': '.priceValue span'}
}
NEWS_RSS = "https://cointelegraph.com/rss"
CACHE_DIR = "./data_cache/"
UK_TIMEZONE = pytz.timezone('Europe/London')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# ----------------------
# Enhanced Data Layer
# ----------------------
@st.cache_data(ttl=300)
def get_multisource_price(pair):
    """Get price from 2 sources with fallback"""
    price1 = scrape_coinmarketcap(pair)
    time.sleep(1.5)  # Rate limiting
    price2 = scrape_coingecko(pair)
    
    if price1 and price2:
        return round((price1 + price2)/2, 2)
    return price1 or price2

def scrape_coinmarketcap(pair):
    try:
        config = CRYPTO_PAIRS[pair]
        url = f'https://coinmarketcap.com/currencies/{config["slug"]}/'
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        selector = st.session_state.get(f'selector_{pair}', config['selector'])
        price_element = soup.select_one(selector)
        
        return float(price_element.text.strip('£').replace(',', '')) if price_element else None
    except Exception as e:
        st.error(f"CMC Error: {str(e)}")
        return None

def scrape_coingecko(pair):
    try:
        coin_id = CRYPTO_PAIRS[pair]['slug']
        url = f'https://www.coingecko.com/en/coins/{coin_id}'
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        price_element = soup.select_one('[data-coin-symbol] .no-wrap')
        return float(price_element.text.split('£')[1].replace(',', '')) if price_element else None
    except Exception as e:
        st.error(f"Coingecko Error: {str(e)}")
        return None

# ----------------------
# Strategy & Risk Management 
# ----------------------
def calculate_dynamic_levels(data):
    """Volatility-adjusted using numpy 1.26+ compatible code"""
    highs = data['High'].iloc[-10:-1].values
    lows = data['Low'].iloc[-10:-1].values
    
    atr = np.nanmax(highs) - np.nanmin(lows)
    volatility_ratio = atr / data['Close'].iloc[-20]
    
    base_sl = 0.25 if volatility_ratio < 0.02 else 0.35
    base_tp = 0.5 if volatility_ratio < 0.02 else 0.75
    
    return {
        'stop_loss': np.nanmin(lows) - (atr * base_sl),
        'take_profit': np.nanmax(highs) + (atr * base_tp),
        'volatility': round(volatility_ratio*100, 2)
    }

# ----------------------
# Performance Monitoring
# ----------------------
def log_signal(pair, entry, sl, tp):
    """Numpy 1.26+ compatible logging"""
    log_entry = {
        'timestamp': datetime.now(UK_TIMEZONE).isoformat(),
        'pair': pair,
        'entry': np.float64(entry),
        'sl': np.float64(sl),
        'tp': np.float64(tp),
        'outcome': 'Pending'
    }
    
    log_path = os.path.join(CACHE_DIR, "performance_log.csv")
    pd.DataFrame([log_entry]).to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

# ----------------------
# Main Application
# ----------------------
def main():
    st.set_page_config(page_title="Pro Crypto Trader", layout="wide")
    
    # Sidebar Controls
    with st.sidebar:
        st.header("Configuration")
        pair = st.selectbox("Pair", list(CRYPTO_PAIRS.keys()))
        account_size = st.number_input("Account Balance (£)", 100, 1000000, 1000)
        risk_percent = st.slider("Risk Percentage", 1, 10, 2)
        
    # Main Dashboard
    col1, col2, col3 = st.columns([1,2,1])
    
    with col1:
        st.header("Live Data")
        current_price = get_multisource_price(pair)
        if current_price:
            st.metric("Current Price", f"£{current_price:,.2f}")
        else:
            st.error("Price retrieval failed")
    
    with col2:
        if current_price:
            data = pd.DataFrame({
                'Close': np.cumsum(np.random.normal(0, 100, 100)) + 30000,
                'High': np.cumsum(np.random.normal(0, 150, 100)) + 30500,
                'Low': np.cumsum(np.random.normal(0, 150, 100)) + 29500
            })
            levels = calculate_dynamic_levels(data)
            
            log_signal(pair, current_price, levels['stop_loss'], levels['take_profit'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price'))
            fig.update_layout(title="Price Action Analysis")
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.header("Risk Management")
        if current_price:
            st.write(f"Volatility: {levels['volatility']}%")
            st.write(f"Stop Loss: £{levels['stop_loss']:,.2f}")
            st.write(f"Take Profit: £{levels['take_profit']:,.2f}")

if __name__ == "__main__":
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    main()
