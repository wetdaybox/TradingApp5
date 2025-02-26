import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime, timedelta
import os

# Configuration
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'BNB-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')
PERFORMANCE_LOG = "trading_performance.csv"

@st.cache_data(ttl=300)
def get_enhanced_data(pair, period='1d', interval='15m'):
    """Improved data fetching with better error handling"""
    try:
        ticker = yf.Ticker(pair)
        data = ticker.history(period=period, interval=interval)
        if data.empty:
            st.error(f"No data found for {pair}")
            return None
        return data
    except Exception as e:
        st.error(f"Data Error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Pro Crypto Trader", layout="wide")
    
    st.title("🚀 Enhanced Crypto Trading Bot")
    st.write("### Advanced Trading Analytics")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Balance (£):", 
                                      min_value=100, 
                                      max_value=1000000, 
                                      value=1000,
                                      step=500)
        risk_percent = st.slider("Risk Percentage:", 
                                min_value=1, 
                                max_value=10, 
                                value=2,
                                help="Percentage of account to risk per trade")
    
    with col2:
        data = get_enhanced_data(pair)
        
        if data is not None:
            current_price = data['Close'].iloc[-1]
            
            # Display basic price info first
            st.subheader(f"Real-Time Data for {pair}")
            st.metric("Current Price", f"£{current_price:,.4f}")
            
            # Price Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='Price'))
            
            fig.update_layout(title=f"{pair} Price Action",
                            xaxis_title="Time",
                            yaxis_title="Price (£)",
                            height=500)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
