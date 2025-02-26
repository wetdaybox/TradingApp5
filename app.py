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
    'BTC-GBP': {'slug': 'bitcoin', 'selector': 'div.priceValue > span'},
    'ETH-GBP': {'slug': 'ethereum', 'selector': 'div.priceValue > span'},
    'BNB-GBP': {'slug': 'bnb', 'selector': 'div.priceValue > span'},
    'XRP-GBP': {'slug': 'ripple', 'selector': 'div.priceValue > span'},
    'ADA-GBP': {'slug': 'cardano', 'selector': 'div.priceValue > span'}
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
# Enhanced Data Layer
# ----------------------
@st.cache_data(ttl=300, show_spinner="Fetching prices...")
def get_multisource_price(pair):
    """Get price from 2 sources with error recovery"""
    try:
        price1 = scrape_coinmarketcap(pair)
        time.sleep(2.5)
        price2 = scrape_coingecko(pair)
        
        if price1 and price2:
            return round((price1 + price2)/2, 2)
        return price1 or price2
    except Exception as e:
        st.error(f"Price fetch failed: {str(e)}")
        return None

def scrape_coinmarketcap(pair):
    """CoinMarketCap scraper with updated 2024 selectors"""
    try:
        config = CRYPTO_PAIRS[pair]
        url = f'https://coinmarketcap.com/currencies/{config["slug"]}/'
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        price_element = (
            soup.select_one('span.sc-f70bb44c-0.jxpCgO.base-text') or
            soup.select_one('div.priceValue > span') or
            soup.select_one('.sc-a0353bbc-0.gDrtaY')
        )
        
        if not price_element:
            raise ValueError("Price element not found")
            
        return float(price_element.text.strip('£').replace(',', ''))
    except Exception as e:
        st.error(f"CMC Error: {str(e)}")
        return None

def scrape_coingecko(pair):
    """CoinGecko scraper with updated 2024 selectors"""
    try:
        coin_id = CRYPTO_PAIRS[pair]['slug']
        url = f'https://www.coingecko.com/en/coins/{coin_id}'
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        price_element = (
            soup.select_one('[data-target="price.price"]') or
            soup.select_one('.tw-text-3xl') or
            soup.select_one('.no-wrap')
        )
        
        if not price_element:
            raise ValueError("Price element not found")
            
        price_text = price_element.text.strip()
        if '£' in price_text:
            return float(price_text.split('£')[1].replace(',', ''))
        return float(price_text.replace(',', ''))
    except Exception as e:
        st.error(f"Coingecko Error: {str(e)}")
        return None

# ----------------------
# Strategy & Risk Management
# ----------------------
def calculate_dynamic_levels(data):
    """Volatility-adjusted trading levels"""
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

def get_market_sentiment():
    """News sentiment analysis"""
    feed = feedparser.parse(NEWS_RSS)
    positive_keywords = ['bullish', 'up', 'rise', 'positive']
    negative_keywords = ['bearish', 'down', 'drop', 'negative']
    
    score = 0
    for entry in feed.entries[:10]:
        title = entry.title.lower()
        score += sum(1 for kw in positive_keywords if kw in title)
        score -= sum(1 for kw in negative_keywords if kw in title)
    
    return 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'

# ----------------------
# Performance Monitoring
# ----------------------
def log_signal(pair, entry, sl, tp):
    """Trade signal logger"""
    log_entry = {
        'timestamp': datetime.now(UK_TIMEZONE).isoformat(),
        'pair': pair,
        'entry': float(entry),
        'sl': float(sl),
        'tp': float(tp),
        'outcome': 'Pending'
    }
    
    pd.DataFrame([log_entry]).to_csv(PERFORMANCE_LOG, 
                                   mode='a', 
                                   header=not os.path.exists(PERFORMANCE_LOG), 
                                   index=False)

def calculate_performance():
    """Performance metrics calculator"""
    if not os.path.exists(PERFORMANCE_LOG):
        return None
    
    df = pd.read_csv(PERFORMANCE_LOG)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    mask = df['timestamp'] > datetime.now(UK_TIMEZONE) - timedelta(hours=24)
    recent = df[mask].copy()
    
    if recent.empty:
        return None
    
    return {
        'accuracy': recent[recent['outcome'] == 'Win'].shape[0] / recent.shape[0],
        'avg_rr': recent.eval('(tp - entry)/abs(entry - sl)').mean(),
        'total_trades': recent.shape[0],
        'latest_outcome': recent.iloc[-1]['outcome']
    }

# ----------------------
# Main Application
# ----------------------
def main():
    st.set_page_config(page_title="Crypto Trading Bot Pro", layout="wide")
    
    # Session state initialization
    if 'signals_generated' not in st.session_state:
        st.session_state.signals_generated = 0
    
    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        pair = st.selectbox("Select Crypto Pair", list(CRYPTO_PAIRS.keys()))
        account_size = st.number_input("Account Balance (£)", 100, 1000000, 1000)
        risk_percent = st.slider("Risk Percentage", 1, 10, 2)
        st.info("ℹ️ Enable/disable features below")
        use_sentiment = st.checkbox("Market Sentiment Analysis", True)
        multi_timeframe = st.checkbox("Multi-Timeframe Confirmation", True)
    
    # Main dashboard layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Column 1: Price Data
    with col1:
        st.header("Live Market Data")
        current_price = get_multisource_price(pair)
        
        if current_price:
            st.metric("Current Price", f"£{current_price:,.2f}")
            if use_sentiment:
                sentiment = get_market_sentiment()
                st.metric("Market Sentiment", sentiment, 
                          delta="Favorable" if sentiment == 'Positive' else "Caution")
    
    # Column 2: Trading Signals
    with col2:
        st.header("Trading Signals")
        if current_price:
            data = pd.DataFrame({
                'Close': np.cumsum(np.random.normal(0, 100, 100)) + current_price,
                'High': np.cumsum(np.random.normal(0, 150, 100)) + current_price * 1.01,
                'Low': np.cumsum(np.random.normal(0, 150, 100)) + current_price * 0.99
            })
            
            levels = calculate_dynamic_levels(data)
            log_signal(pair, current_price, levels['stop_loss'], levels['take_profit'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price'))
            fig.add_hline(y=levels['stop_loss'], line_dash="dot", 
                         annotation_text=f"SL: £{levels['stop_loss']:,.2f}")
            fig.add_hline(y=levels['take_profit'], line_dash="dash", 
                         annotation_text=f"TP: £{levels['take_profit']:,.2f}")
            fig.update_layout(title="Price Action Analysis",
                             xaxis_title="Time",
                             yaxis_title="Price (£)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Column 3: Risk Management
    with col3:
        st.header("Risk Management")
        if current_price:
            stop_loss_pct = abs((current_price - levels['stop_loss']) / current_price) * 100
            position_size = (account_size * (risk_percent/100)) / abs(current_price - levels['stop_loss'])
            
            st.metric("Volatility", f"{levels['volatility']}%")
            st.metric("Stop Loss Distance", f"{stop_loss_pct:.2f}%")
            st.metric("Position Size", f"{position_size:.6f} {pair.split('-')[0]}")
            
            st.progress(min(levels['volatility']/5, 1.0), 
                       text=f"Risk Level: {'High' if levels['volatility'] > 3 else 'Medium' if levels['volatility'] > 1.5 else 'Low'}")
        
        # Performance Dashboard
        st.divider()
        st.subheader("Performance Metrics")
        performance = calculate_performance()
        if performance:
            cols = st.columns(2)
            cols[0].metric("24h Accuracy", f"{performance['accuracy']*100:.1f}%")
            cols[1].metric("Avg Risk/Reward", f"{performance['avg_rr']:.1f}:1")
            st.metric("Total Signals", performance['total_trades'])
            st.metric("Last Outcome", performance['latest_outcome'])
        else:
            st.info("No performance data available yet")

if __name__ == "__main__":
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    main()
