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
            data['RSI'] = calculate_rsi(data)
            st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        return data
    except Exception as e:
        st.error(f"⚠️ Data error: {str(e)}")
        return pd.DataFrame()

def calculate_rsi(data, window=14):
    """Calculate RSI with error handling"""
    try:
        close_prices = data['Close'] if 'Close' in data else data['Adj Close']
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        st.error(f"⚠️ RSI Calculation Error: {str(e)}")
        return pd.Series([50]*len(data))  # Default neutral value

def main():
    st.set_page_config(page_title="Crypto Trader Pro+", layout="wide")
    st.title("🚀 Smart Crypto Trading Assistant")
    
    # Show loading spinner while initializing
    with st.spinner('Initializing trading engine...'):
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
            st.header("📊 Live Market Analysis")
            
            try:
                # Get market data
                data = get_realtime_data(pair)
                
                if data.empty:
                    st.warning("📡 Connecting to market data...")
                    return
                
                # Get current price
                current_price = get_current_price(data)
                if not current_price:
                    st.warning("⏳ Waiting for price data...")
                    return
                
                # Display interface
                display_trading_interface(data, current_price)
                st.success(f"✅ Updated at {st.session_state.last_update} UTC")
                
            except Exception as e:
                st.error(f"🔥 System Error: {str(e)}")
                st.info("ℹ️ Please refresh or try another asset")

def display_trading_interface(data, current_price):
    """Dynamic trading interface with real calculations"""
    with st.container():
        # Calculate trading parameters
        stop_loss = current_price * 0.95  # 5% stop loss
        take_profit = current_price * 1.15  # 15% target
        rsi_value = data['RSI'].iloc[-1] if 'RSI' in data else 50
        
        # Generate recommendation
        recommendation = "Buy" if rsi_value < RSI_OVERSOLD else \
                       "Sell" if rsi_value > RSI_OVERBOUGHT else "Hold"
        
        # Display metrics
        cols = st.columns(3)
        cols[0].metric("Current Price", f"£{current_price:,.4f}")
        cols[1].metric("24h Range", 
                      f"£{data['Low'].min():,.4f}-£{data['High'].max():,.4f}")
        cols[2].metric("Volatility", f"±{data['Close'].pct_change().std()*100:.2f}%")
        
        # Trading strategy card
        with st.expander("📈 Trading Strategy", expanded=True):
            st.write(f"""
            **Recommended Action:** {recommendation}  
            **Target Price:** £{take_profit:,.4f} (+15%)  
            **Stop Loss:** £{stop_loss:,.4f} (-5%)
            **RSI Indicator:** {rsi_value:.1f} ({get_rsi_status(rsi_value)})
            """)
        
        # Price chart
        plot_price_chart(data)

def get_rsi_status(rsi_value):
    """Get RSI status description"""
    if rsi_value < RSI_OVERSOLD:
        return "Oversold 🟢"
    if rsi_value > RSI_OVERBOUGHT:
        return "Overbought 🔴"
    return "Neutral ⚪"

def plot_price_chart(data):
    """Interactive price chart with error handling"""
    try:
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'] if 'Close' in data else data['Adj Close']
        )])
        fig.update_layout(
            height=500,
            title="Live Price Chart",
            yaxis_title="Price (£)",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"📉 Chart Error: {str(e)}")

def get_current_price(data):
    """Get current price with error handling"""
    try:
        close_prices = data['Close'] if 'Close' in data else data['Adj Close']
        fx_rate = get_fx_rate()
        return close_prices.iloc[-1].item() / fx_rate
    except Exception as e:
        st.error(f"💱 Price Error: {str(e)}")
        return None

@st.cache_data(ttl=60)
def get_fx_rate():
    """Get GBP/USD rate with error handling"""
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='5m', auto_adjust=True)
        return fx_data['Close'].iloc[-1].item() if not fx_data.empty else 0.80
    except Exception as e:
        st.error(f"💱 FX Error: {str(e)}")
        return 0.80

if __name__ == "__main__":
    main()
