import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
import sqlite3
from datetime import datetime, timedelta

# 🇬🇧 British Configuration 🇬🇧
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'BNB-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')

# Database setup
conn = sqlite3.connect('trading_journal.db')
c = conn.cursor()

def get_proper_price(data):
    """Ensure we get a proper float value"""
    try:
        return float(data['Close'].iloc[-1].item())  # Convert to native Python float
    except:
        return None

@st.cache_data(ttl=30)
def get_realtime_data(pair):
    """Robust data fetching with type conversion"""
    try:
        data = yf.download(pair, period='2d', interval='5m', progress=False, auto_adjust=True)
        if not data.empty:
            data.index = data.index.tz_convert(UK_TIMEZONE)
            data['Close'] = data['Close'].astype(float)  # Ensure float type
            return data
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def generate_trade_advice(signals, current_price):
    """Safe string formatting"""
    advice = []
    try:
        # Convert numpy types to native Python types
        current_price = float(current_price)
        advice.append(f"💷 Current Price: £{current_price:,.2f}")
    except Exception as e:
        st.error(f"Price formatting error: {str(e)}")
    
    # Rest of advice generation remains same
    return advice

def main():
    st.set_page_config(page_title="🇬🇧 Crypto Trader Pro", layout="centered")
    st.title("🇬🇧 Personal Trading Assistant")
    
    pair = st.selectbox("Select Asset:", CRYPTO_PAIRS)
    data = get_realtime_data(pair)
    
    if not data.empty:
        current_price = get_proper_price(data)
        if current_price:
            # Rest of your original logic
            st.subheader(f"Current Price: £{current_price:,.2f}")
        else:
            st.warning("Price data unavailable")
    else:
        st.warning("Loading market data...")

if __name__ == "__main__":
    main()
