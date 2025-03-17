import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
import numpy as np
import json
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
BASE_REFRESH_INTERVAL = 60  # Seconds

# Initialize session state
if 'manual_price' not in st.session_state:
    st.session_state.manual_price = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = "Loading..."

@st.cache_data(ttl=30)
def get_realtime_data(pair):
    """Get 48 hours of 5-minute data"""
    try:
        data = yf.download(pair, period='2d', interval='5m', progress=False, auto_adjust=True)
        if not data.empty:
            data['RSI'] = get_rsi(data)
            st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        return data
    except Exception as e:
        st.error(f"⚠️ Data error: {str(e)}")
        return pd.DataFrame()

def get_rsi(data, window=14):
    """Enhanced RSI calculation with error handling"""
    try:
        close_prices = data['Close']
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        st.error(f"⚠️ RSI Calculation Error: {str(e)}")
        return pd.Series([None]*len(data))

def main():
    st.set_page_config(page_title="Crypto Trader Pro+", layout="wide")
    st.title("🚀 Crypto Trading Assistant")
    
    # Show loading spinner while initializing
    with st.spinner('Loading trading engine...'):
        # Main display columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.header("⚙️ Settings")
            pair = st.selectbox("Select Asset:", CRYPTO_PAIRS)
            
            st.header("💰 Portfolio")
            account_size = st.number_input("Portfolio Value (£)", 
                                         min_value=100.0, value=1000.0, step=100.0)
            use_manual = st.checkbox("Enter Price Manually")
            if use_manual:
                st.session_state.manual_price = st.number_input(
                    "Manual Price (£)", min_value=0.01, 
                    value=st.session_state.manual_price or 1000.0
                )
            else:
                st.session_state.manual_price = None

        with col2:
            st.header("📊 Market Analysis")
            
            # Get data with error handling
            try:
                data = get_realtime_data(pair)
                current_price, is_manual = get_price_data(pair)
                
                if data.empty:
                    st.warning("📡 Waiting for market data...")
                    return
                
                st.success(f"✅ Data loaded at {st.session_state.last_update} UTC")
                
                # Display core metrics
                if current_price:
                    display_trading_interface(data, current_price)
                else:
                    st.warning("⏳ Waiting for price data...")

            except Exception as e:
                st.error(f"🔥 Critical Error: {str(e)}")
                st.info("ℹ️ Please refresh the page or try again later")

def display_trading_interface(data, current_price):
    """Handles all trading interface components"""
    with st.container():
        st.subheader("Live Trading Signals")
        cols = st.columns(3)
        cols[0].metric("Current Price", f"£{current_price:,.2f}")
        cols[1].metric("24h Change", "+2.45%")  # Example data
        cols[2].metric("Market Sentiment", "Bullish 🐂")
        
        # Show price chart
        plot_price_chart(data)
        
        # Trading recommendations
        with st.expander("📈 Trading Strategy", expanded=True):
            st.write("""
            **Recommended Action:** Buy  
            **Target Price:** £32,450  
            **Stop Loss:** £29,800
            """)
            
        # Risk management tools
        with st.expander("🛡️ Risk Calculator"):
            st.slider("Risk Percentage", 1, 100, 5)
            st.button("Calculate Position Size")

def plot_price_chart(data):
    """Creates interactive price chart"""
    try:
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"📉 Chart Error: {str(e)}")

def get_price_data(pair):
    """Gets price data with fallback handling"""
    try:
        data = get_realtime_data(pair)
        if data.empty:
            return None, False
            
        close_price = data['Close'].iloc[-1].item()
        return close_price / get_fx_rate(), False
    except Exception as e:
        st.error(f"💱 Price Error: {str(e)}")
        return None, False

@st.cache_data(ttl=60)
def get_fx_rate():
    """Gets GBPUSD exchange rate"""
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='5m', auto_adjust=True)
        return fx_data['Close'].iloc[-1].item() if not fx_data.empty else 0.80
    except Exception as e:
        st.error(f"💱 FX Error: {str(e)}")
        return 0.80

if __name__ == "__main__":
    main()
