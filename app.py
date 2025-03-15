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
        return float(data['Close'].iloc[-1]) if not data.empty else None
    except Exception as e:
        st.error(f"Error fetching price: {e}")
        return None

@st.cache_data(ttl=300)
def download_data(pair, period='1d', interval='15m'):
    """Download historical data"""
    return yf.download(pair, period=period, interval=interval)

def calculate_levels(pair):
    """Calculate trading levels"""
    data = download_data(pair)
    if data.empty or len(data) < 20:
        return None
    
    high = data['High'].iloc[-20:-1].max()
    low = data['Low'].iloc[-20:-1].min()
    current_price = data['Close'].iloc[-1]
    
    return {
        'buy_zone': round((high + low) / 2, 2),
        'take_profit': round(high + (high - low) * 0.5, 2),
        'stop_loss': round(low - (high - low) * 0.25, 2),
        'current': current_price
    }

def calculate_position_size(account_size, risk_percent, stop_loss_distance):
    """Risk management calculator"""
    if stop_loss_distance <= 0:
        return 0
    risk_amount = account_size * (risk_percent / 100)
    return round(risk_amount / stop_loss_distance, 4)

def main():
    st.set_page_config(page_title="Crypto Trader", layout="centered")
    st.title("🇬🇧 Free Crypto Trading Bot")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Balance (£):", 
                                     min_value=100, value=1000)
        risk_percent = st.slider("Risk Percentage:", 1, 10, 2)
    
    with col2:
        current_price = get_realtime_price(pair)
        if current_price:
            levels = calculate_levels(pair)
            if levels:
                stop_loss_distance = abs(current_price - levels['stop_loss'])
                position_size = calculate_position_size(account_size, risk_percent, stop_loss_distance)
                
                st.write("## Trading Signals")
                st.metric("Current Price", f"£{current_price:,.2f}")
                st.write(f"**Buy Zone:** £{levels['buy_zone']:,.2f}")
                st.write(f"**Take Profit:** £{levels['take_profit']:,.2f}")
                st.write(f"**Stop Loss:** £{levels['stop_loss']:,.2f}")
                st.write(f"**Position Size:** {position_size:,.4f} {pair.split('-')[0]}")

if __name__ == "__main__":
    main()
