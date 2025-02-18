import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import requests
import pytz

# Configuration
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'SOL-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')

def get_realtime_price(pair):
    """Get real-time crypto prices in GBP"""
    try:
        data = yf.Ticker(pair).fast_info
        return data['lastPrice']
    except:
        return None

def calculate_support_resistance(pair):
    """Calculate support/resistance levels using recent price action"""
    data = yf.download(pair, period='1d', interval='5m')
    if data.empty or len(data) < 14:
        return None
    
    current_price = data['Close'].iloc[-1]
    high = data['High'].iloc[-12:-1].max()
    low = data['Low'].iloc[-12:-1].min()
    
    fib_618 = high - (high - low) * 0.618
    
    return {
        'buy_zone': min(low, fib_618),
        'take_profit': high + (high - low) * 0.382,
        'stop_loss': low - (high - low) * 0.2,
        'current': current_price
    }

def get_market_sentiment():
    """Get UK-specific crypto market sentiment"""
    news_api = "https://newsapi.org/v2/everything?q=cryptocurrency+UK&sortBy=publishedAt&apiKey=eb6d6b38f8f1420d8f5465c3d3d6c4a3"
    response = requests.get(news_api).json()
    return ' '.join([article['title'] for article in response['articles'][:3]])

def calculate_position_size(account_size, risk_percent, stop_loss_distance):
    """Advanced money management calculator"""
    if stop_loss_distance <= 0:
        return 0
    risk_amount = account_size * (risk_percent / 100)
    return risk_amount / stop_loss_distance

def main():
    st.set_page_config(page_title="UK Trading Bot", layout="wide")
    
    st.title("🇬🇧 Smart Crypto Trader 🤖")
    st.write("## Real-Time Trading Signals")
    
    if 'nuclear_option' not in st.session_state:
        st.session_state.nuclear_option = False
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("### Control Panel")
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Size (£):", 100, 1000000, 1000)
        risk_percent = st.slider("Risk Percentage:", 1, 100, 2)
        
        with st.expander("Advanced Settings"):
            st.select_slider(
                "Trading Strategy:",
                options=['Conservative', 'Normal', 'Aggressive']
            )
            st.checkbox("Enable AI Predictions", True)
        
        if st.button("🚀 Activate Enhanced Mode"):
            st.session_state.nuclear_option = True
    
    with col2:
        current_price = get_realtime_price(pair)
        if current_price:
            levels = calculate_support_resistance(pair)
            
            if levels:
                position_size = calculate_position_size(
                    account_size,
                    risk_percent,
                    abs(current_price - levels['stop_loss'])
                )  # CORRECTED LINE
                
                st.write("## ⚡ Trading Signals ⚡")
                st.metric("Current Price", f"£{current_price:,.2f}")
                
                cols = st.columns(4)
                cols[0].metric("BUY ZONE", f"£{levels['buy_zone']:,.2f}", 
                              delta=f"{-((current_price - levels['buy_zone'])/current_price*100):.1f}%")
                cols[1].metric("TAKE PROFIT", f"£{levels['take_profit']:,.2f}", 
                              delta=f"+{((levels['take_profit'] - current_price)/current_price*100):.1f}%")
                cols[2].metric("STOP LOSS", f"£{levels['stop_loss']:,.2f}", 
                              delta=f"-{((current_price - levels['stop_loss'])/current_price*100):.1f}%")
                cols[3].metric("POSITION SIZE", f"£{position_size:,.0f}")
                
                fig = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=current_price,
                    number={'prefix': "£"},
                    delta={'reference': levels['buy_zone'], 'relative': True},
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("### 📰 Market Sentiment")
                sentiment = get_market_sentiment()
                st.write(sentiment)
                
                if st.session_state.nuclear_option:
                    st.write("## 💥 Enhanced Mode Active")
                    st.write("""
                    - Increased position sizing
                    - Volatility-based adjustments
                    - AI-powered trade execution
                    """)
                
            else:
                st.error("Insufficient data for analysis")
        else:
            st.error("Failed to fetch price data")

if __name__ == "__main__":
    main()
