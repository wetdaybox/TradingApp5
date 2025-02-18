import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import backtrader as bt
import requests
import pytz
from datetime import datetime, timedelta
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import plotly.graph_objects as go
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'XRP-USD']
STRATEGIES = ['SMA Crossover', 'RSI Divergence', 'MACD Momentum']
RISK_PARAMS = {
    'max_risk_per_trade': 0.02,
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.10
}

# Initialize session state
if 'bot_state' not in st.session_state:
    st.session_state.bot_state = {
        'positions': {},
        'capital': 10000,
        'historical_data': {},
        'performance': pd.DataFrame(),
        'last_update': datetime.now(pytz.utc)
    }

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_market_data(pair):
    """Multi-source data fetching with enhanced error handling"""
    try:
        data = yf.download(pair, period='1d', interval='5m', progress=False)
        if not data.empty:
            return data
    except Exception as e:
        st.warning(f"Yahoo Finance failed: {str(e)}")
    
    try:
        coin_id = pair.split("-")[0].lower()
        response = requests.get(
            f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=1',
            timeout=10
        )
        response.raise_for_status()
        ohlc = response.json()
        if isinstance(ohlc, list) and len(ohlc) > 0:
            df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.set_index('timestamp').sort_index()
    except Exception as e:
        st.error(f"CoinGecko failed: {str(e)}")
    
    return pd.DataFrame()

def update_market_data():
    """Enhanced data update with status tracking"""
    for pair in CRYPTO_PAIRS:
        data = fetch_market_data(pair)
        if not data.empty:
            st.session_state.bot_state['historical_data'][pair] = data
            st.success(f"Updated {pair} data")
        else:
            st.error(f"Failed to update {pair} data")

# Rest of the code remains the same as previous version except positions display:
    
with col1:
    st.metric("Available Capital", f"${st.session_state.bot_state['capital']:,.2f}")
    st.metric("Open Positions", len(st.session_state.bot_state['positions']))
    
    if st.session_state.bot_state['positions']:
        st.write("### Current Positions")
        for pair, position in st.session_state.bot_state['positions'].items():
            try:
                current_price = yf.Ticker(pair).history(period='1d').iloc[-1]['Close']
            except:
                # Fallback to CoinGecko
                coin_id = pair.split("-")[0].lower()
                response = requests.get(
                    f'https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd',
                    timeout=5
                )
                current_price = response.json().get(coin_id, {}).get('usd', position['entry_price'])
            
            pnl = (current_price / position['entry_price'] - 1) * 100
            st.write(f"""
            **{pair}**  
            Quantity: {position['quantity']:.4f}  
            Entry: ${position['entry_price']:.2f}  
            Current: ${current_price:.2f} ({pnl:.2f}%)  
            SL: ${position['stop_loss']:.2f}  
            TP: ${position['take_profit']:.2f}
            """)
