# app.py
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
    'BTC-GBP': {'slug': 'bitcoin', 'selectors': ['span.sc-f70bb44c-0', 'div.priceValue']},
    'ETH-GBP': {'slug': 'ethereum', 'selectors': ['span.sc-f70bb44c-0', 'div.priceValue']},
    'BNB-GBP': {'slug': 'bnb', 'selectors': ['span.sc-f70bb44c-0', 'div.priceValue']},
    'XRP-GBP': {'slug': 'ripple', 'selectors': ['span.sc-f70bb44c-0', 'div.priceValue']},
    'ADA-GBP': {'slug': 'cardano', 'selectors': ['span.sc-f70bb44c-0', 'div.priceValue']}
}
NEWS_RSS = "https://cointelegraph.com/rss"
CACHE_DIR = "./data_cache/"
PERFORMANCE_LOG = os.path.join(CACHE_DIR, "performance_log.csv")
UK_TIMEZONE = pytz.timezone('Europe/London')
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept-Language': 'en-GB,en;q=0.9',
    'Referer': 'https://www.google.com/'
}

# ----------------------
# Data Layer Functions
# ----------------------
@st.cache_data(ttl=300, show_spinner="Fetching prices...")
def get_multisource_price(pair):
    """Get price from multiple sources with fallback"""
    sources = [scrape_coinmarketcap, scrape_coingecko, scrape_coindesk]
    prices = []
    
    for source in sources:
        try:
            if price := source(pair):
                prices.append(price)
                if len(prices) >= 2:
                    break
            time.sleep(1.5)
        except Exception:
            continue
            
    return np.mean(prices) if prices else None

def scrape_coinmarketcap(pair):
    """CoinMarketCap scraper with multiple fallback selectors"""
    try:
        config = CRYPTO_PAIRS[pair]
        url = f'https://coinmarketcap.com/currencies/{config["slug"]}/'
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for selector in config['selectors']:
            if element := soup.select_one(selector):
                return float(element.text.strip('£').replace(',', ''))
        return None
    except Exception as e:
        st.error(f"CMC Error: {str(e)}")
        return None

def scrape_coingecko(pair):
    """CoinGecko scraper with updated selectors"""
    try:
        coin_id = CRYPTO_PAIRS[pair]['slug']
        url = f'https://www.coingecko.com/en/coins/{coin_id}'
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if element := soup.select_one('[data-target="price.price"]'):
            return float(element.text.split('£')[1].replace(',', ''))
        return None
    except Exception as e:
        st.error(f"Gecko Error: {str(e)}")
        return None

def scrape_coindesk(pair):
    """Fallback price source"""
    try:
        coin = pair.split('-')[0]
        url = f'https://www.coindesk.com/price/{coin}-gbp/'
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if element := soup.select_one('div.price-large'):
            return float(element.text.strip().replace('£', '').replace(',', ''))
        return None
    except Exception:
        return None

# ----------------------
# Strategy Components
# ----------------------
def calculate_dynamic_levels(data):
    """Calculate volatility-adjusted trading levels"""
    highs = data['High'].iloc[-10:-1].values
    lows = data['Low'].iloc[-10:-1].values
    
    atr = np.nanmax(highs) - np.nanmin(lows)
    volatility = atr / data['Close'].iloc[-20]
    
    return {
        'stop_loss': np.nanmin(lows) - (atr * 0.3),
        'take_profit': np.nanmax(highs) + (atr * 0.6),
        'volatility': round(volatility * 100, 2)
    }

def get_market_sentiment():
    """Analyze market sentiment from RSS feed"""
    feed = feedparser.parse(NEWS_RSS)
    positive = ['bullish', 'up', 'rise', 'positive']
    negative = ['bearish', 'down', 'drop', 'negative']
    score = 0
    
    for entry in feed.entries[:10]:
        title = entry.title.lower()
        score += sum(1 for kw in positive if kw in title)
        score -= sum(1 for kw in negative if kw in title)
    
    return 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'

# ----------------------
# Performance Tracking
# ----------------------
def log_signal(pair, entry, sl, tp):
    """Log trading signals"""
    log_entry = {
        'timestamp': datetime.now(UK_TIMEZONE).isoformat(),
        'pair': pair,
        'entry': entry,
        'sl': sl,
        'tp': tp,
        'outcome': 'Pending'
    }
    pd.DataFrame([log_entry]).to_csv(PERFORMANCE_LOG, mode='a', header=not os.path.exists(PERFORMANCE_LOG), index=False)

def calculate_performance():
    """Calculate performance metrics"""
    if not os.path.exists(PERFORMANCE_LOG):
        return None
    
    df = pd.read_csv(PERFORMANCE_LOG)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    recent = df[df['timestamp'] > datetime.now(UK_TIMEZONE) - timedelta(hours=24)]
    
    if recent.empty:
        return None
    
    return {
        'accuracy': recent[recent['outcome'] == 'Win'].shape[0] / len(recent),
        'avg_rr': recent.eval('(tp-entry)/abs(entry-sl)').mean(),
        'total_trades': len(recent),
        'latest_outcome': recent.iloc[-1]['outcome']
    }

# ----------------------
# Main Application
# ----------------------
def main():
    st.set_page_config(page_title="Crypto Trading Bot", layout="wide")
    
    with st.sidebar:
        st.header("Configuration")
        pair = st.selectbox("Select Pair", list(CRYPTO_PAIRS.keys()))
        balance = st.number_input("Account Balance (£)", 100, 1000000, 1000)
        risk = st.slider("Risk %", 1, 10, 2)
        use_sentiment = st.checkbox("Enable Sentiment", True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.header("Market Data")
        price = get_multisource_price(pair)
        
        if price:
            st.metric("Price", f"£{price:,.2f}")
            if use_sentiment:
                sentiment = get_market_sentiment()
                st.metric("Sentiment", sentiment, 
                         delta="Favorable" if sentiment == 'Positive' else "Caution")
    
    with col2:
        if price:
            data = pd.DataFrame({
                'Close': np.cumsum(np.random.normal(0, 100, 100)) + price,
                'High': np.cumsum(np.random.normal(0, 150, 100)) + price * 1.02,
                'Low': np.cumsum(np.random.normal(0, 150, 100)) + price * 0.98
            })
            levels = calculate_dynamic_levels(data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price'))
            fig.add_hline(y=levels['stop_loss'], line_dash="dot", 
                         annotation_text=f"SL: £{levels['stop_loss']:,.2f}")
            fig.add_hline(y=levels['take_profit'], line_dash="dash", 
                         annotation_text=f"TP: £{levels['take_profit']:,.2f}")
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        if price:
            stop_pct = abs((price - levels['stop_loss']) / price) * 100
            size = (balance * (risk/100)) / abs(price - levels['stop_loss'])
            
            st.metric("Volatility", f"{levels['volatility']}%")
            st.metric("Stop Distance", f"{stop_pct:.2f}%")
            st.metric("Position Size", f"{size:.6f} {pair.split('-')[0]}")
            
            st.progress(min(levels['volatility']/5, 1.0), 
                       text=f"Risk: {'High' if levels['volatility'] > 3 else 'Medium' if levels['volatility'] > 1.5 else 'Low'}")

if __name__ == "__main__":
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    main()
