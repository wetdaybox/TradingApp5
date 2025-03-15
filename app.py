import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime

# Configuration
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'BNB-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')

@st.cache_data(ttl=300)
def get_realtime_price(pair):
    """Get real-time crypto prices in GBP"""
    try:
        data = yf.Ticker(pair).history(period='1d', interval='1m')
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return None
    except Exception as e:
        st.error(f"Price error: {str(e)}")
        return None

@st.cache_data(ttl=300)
def download_data(pair, period='1d', interval='15m'):
    """Download historical data with error handling"""
    try:
        return yf.download(pair, period=period, interval=interval, progress=False)
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def calculate_levels(pair):
    """Calculate trading levels with validation"""
    data = download_data(pair)
    if data.empty or len(data) < 20:
        return None
    
    try:
        closed_data = data.iloc[:-1] if len(data) > 1 else data
        high = closed_data['High'].iloc[-20:].max()
        low = closed_data['Low'].iloc[-20:].min()
        current_price = data['Close'].iloc[-1]
        
        # Convert to floats explicitly
        high = float(high)
        low = float(low)
        current_price = float(current_price)
        
        stop_loss = max(0.0, low - (high - low) * 0.25)
        
        return {
            'buy_zone': round((high + low) / 2, 2),
            'take_profit': round(high + (high - low) * 0.5, 2),
            'stop_loss': round(stop_loss, 2),
            'current': current_price
        }
    except Exception as e:
        st.error(f"Level calculation error: {str(e)}")
        return None

def calculate_position_size(account_size, risk_percent, stop_loss_distance):
    """Safe position sizing"""
    try:
        # Ensure numeric value
        stop_loss_distance = float(stop_loss_distance)
        if stop_loss_distance <= 0:
            return 0.0
            
        risk_amount = account_size * (risk_percent / 100)
        return round(risk_amount / stop_loss_distance, 4)
    except Exception as e:
        st.error(f"Position sizing error: {str(e)}")
        return 0.0

# Rest of the functions remain unchanged...

def main():
    st.set_page_config(page_title="Crypto Trader", layout="centered")
    st.title("🇬🇧 Free Crypto Trading Bot")
    st.write("### Risk-Managed Trading Signals")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Balance (£):", 
                                     min_value=100, max_value=1000000, value=1000)
        risk_percent = st.slider("Risk Percentage:", 1, 10, 2)
        base_currency = pair.split('-')[0]
    
    with col2:
        current_price = get_realtime_price(pair)
        if current_price:
            levels = calculate_levels(pair)
            if levels:
                try:
                    stop_loss_distance = abs(current_price - levels['stop_loss'])
                    position_size = calculate_position_size(account_size, risk_percent, stop_loss_distance)
                    notional_value = position_size * current_price
                    
                    # Display section remains unchanged...
                    
                except Exception as e:
                    st.error(f"Display error: {str(e)}")
            else:
                st.error("Insufficient market data for analysis")
        else:
            st.error("Couldn't fetch current prices. Try again later.")

if __name__ == "__main__":
    main()
