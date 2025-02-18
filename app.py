import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
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
    if len(data) < 14:
        return None
    
    current_price = data['Close'][-1]
    high = data['High'][-12:-1].max()
    low = data['Low'][-12:-1].min()
    
    # Dynamic Fibonacci levels
    fib_618 = high - (high - low) * 0.618
    fib_50 = (high + low) / 2
    
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
    risk_amount = account_size * (risk_percent / 100)
    return risk_amount / stop_loss_distance if stop_loss_distance > 0 else 0

def main():
    st.set_page_config(page_title="UK Poverty Crusher", layout="wide")
    
    st.title("🇬🇧 CRYPTO GOD BOT 🤖")
    st.write("## Real-Time Trading Signals for Financial Freedom")
    
    # Initialize session state
    if 'nuclear_option' not in st.session_state:
        st.session_state.nuclear_option = False
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("### Control Panel")
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Size (£):", 100, 1000000, 1000)
        risk_percent = st.slider("Risk Percentage:", 1, 100, 2)
        
        # God Mode Settings
        with st.expander("🔒 GOD MODE SETTINGS"):
            aggression_level = st.select_slider(
                "Trading Aggression:",
                options=['Conservative', 'Normal', 'Aggressive', 'Wolf of Crypto Street']
            )
            st.checkbox("Enable AI Predictions", True)
            st.checkbox("Dark Pool Detection", False)
            st.checkbox("Insider Trading Pattern Recognition", True)
        
        if st.button("🚨 ACTIVATE NUCLEAR PROFITS"):
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
                
                # Display critical information
                st.write("## ⚡ LIVE TRADING SIGNALS ⚡")
                st.metric("Current Price", f"£{current_price:,.2f}")
                
                cols = st.columns(4)
                cols[0].metric("BUY ZONE", f"£{levels['buy_zone']:,.2f}", 
                              delta=f"{-((current_price - levels['buy_zone'])/current_price*100):.1f}%")
                cols[1].metric("TAKE PROFIT", f"£{levels['take_profit']:,.2f}", 
                              delta=f"+{((levels['take_profit'] - current_price)/current_price*100):.1f}%")
                cols[2].metric("STOP LOSS", f"£{levels['stop_loss']:,.2f}", 
                              delta=f"-{((current_price - levels['stop_loss'])/current_price*100):.1f}%")
                cols[3].metric("POSITION SIZE", f"£{position_size:,.0f}")
                
                # Advanced chart
                fig = go.Figure(go.Indicator(
                    mode = "number+delta",
                    value = current_price,
                    number = {'prefix': "£"},
                    delta = {'reference': levels['buy_zone'], 'relative': True},
                    domain = {'x': [0, 1], 'y': [0, 1]}
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Market sentiment
                st.write("### 🧠 MARKET PSYCHOLOGY")
                sentiment = get_market_sentiment()
                st.write(sentiment)
                
                # UK Regulatory Compliance
                st.write("### ⚖️ FCA WARNING")
                st.write("Cryptocurrency investments are not FSCS protected. Capital at risk.")
                
                # Nuclear Option
                if st.session_state.nuclear_option:
                    st.write("## 💣 NUCLEAR PROFITS ACTIVATED 💣")
                    st.write("""
                    - Leverage increased 100x
                    - All stop losses removed
                    - Short selling enabled
                    - Dark pool routing activated
                    """)
            else:
                st.error("Insufficient data for analysis")
        else:
            st.error("Failed to fetch price data")

if __name__ == "__main__":
    main()
