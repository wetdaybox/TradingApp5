# trading_bot.py
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
PERFORMANCE_LOG = os.path.join(CACHE_DIR, "performance_log.csv")
UK_TIMEZONE = pytz.timezone('Europe/London')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# ----------------------
# Enhanced Data Layer
# ----------------------
@st.cache_data(ttl=300, show_spinner="Fetching prices...")
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
    """Volatility-adjusted levels using numpy 1.26+ compatible code"""
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
    """RSS news analysis"""
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
    """Log signal with numpy 1.26+ compatible types"""
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
    """Calculate key metrics from historical log"""
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
        'avg_rr': recent['risk_reward'].mean(),
        'total_trades': recent.shape[0],
        'latest_outcome': recent.iloc[-1]['outcome']
    }

# ----------------------
# Main Application
# ----------------------
def main():
    st.set_page_config(page_title="Crypto Trading Bot Pro", layout="wide")
    
    # Initialize session state
    if 'signals_generated' not in st.session_state:
        st.session_state.signals_generated = 0
    
    # Sidebar Controls
    with st.sidebar:
        st.header("Configuration")
        pair = st.selectbox("Select Crypto Pair", list(CRYPTO_PAIRS.keys()))
        account_size = st.number_input("Account Balance (£)", 100, 1000000, 1000)
        risk_percent = st.slider("Risk Percentage", 1, 10, 2)
        st.info("ℹ️ Enable/disable features below")
        use_sentiment = st.checkbox("Market Sentiment Analysis", True)
        multi_timeframe = st.checkbox("Multi-Timeframe Confirmation", True)
    
    # Main Dashboard
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Price Data Column
    with col1:
        st.header("Live Market Data")
        current_price = get_multisource_price(pair)
        
        if current_price:
            st.metric("Current Price", f"£{current_price:,.2f}")
            sentiment = get_market_sentiment() if use_sentiment else 'Neutral'
            st.metric("Market Sentiment", sentiment, 
                     delta="Favorable" if sentiment == 'Positive' else "Caution")
        else:
            st.error("Price retrieval failed. Try refreshing.")
    
    # Trading Signals Column
    with col2:
        st.header("Trading Signals")
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
            fig.add_hline(y=levels['stop_loss'], line_dash="dot", 
                         annotation_text=f"SL: £{levels['stop_loss']:,.2f}")
            fig.add_hline(y=levels['take_profit'], line_dash="dash", 
                         annotation_text=f"TP: £{levels['take_profit']:,.2f}")
            fig.update_layout(title="Price Action Analysis",
                             xaxis_title="Time",
                             yaxis_title="Price (£)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk Management Column
    with col3:
        st.header("Risk Management")
        if current_price:
            stop_loss_pct = abs((current_price - levels['stop_loss']) / current_price * 100
            position_size = (account_size * (risk_percent/100)) / (current_price - levels['stop_loss'])
            
            st.metric("Volatility", f"{levels['volatility']}%")
            st.metric("Stop Loss Distance", f"{stop_loss_pct:.2f}%")
            st.metric("Position Size", f"{position_size:.4f} {pair.split('-')[0]}")
            
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
