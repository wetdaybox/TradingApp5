import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import backtrader as bt
import requests
import pytz
import time
from datetime import datetime
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
API_COOLDOWN = 65  # Seconds between API calls (CoinGecko limit)

# Initialize session state
if 'bot_state' not in st.session_state:
    st.session_state.update({
        'bot_state': {
            'positions': {},
            'capital': 10000,
            'historical_data': {},
            'performance': pd.DataFrame(),
            'last_update': datetime.now(pytz.utc),
            'last_api_call': 0,
            'update_in_progress': False
        },
        'data_cache': {}
    })

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
@st.cache_data(ttl=600, show_spinner=False)
def fetch_market_data(pair):
    """Multi-source data fetching with strict rate limiting"""
    current_time = time.time()
    
    # Enforce API cooldown
    if current_time - st.session_state.bot_state['last_api_call'] < API_COOLDOWN:
        return pd.DataFrame()
    
    try:
        # Try Yahoo Finance first
        data = yf.download(pair, period='1d', interval='5m', progress=False)
        if not data.empty:
            st.session_state.bot_state['last_api_call'] = time.time()
            return data
    except Exception as e:
        pass
    
    try:
        # Fallback to CoinGecko (public API)
        st.session_state.bot_state['last_api_call'] = time.time()
        coin_id = pair.split("-")[0].lower()
        response = requests.get(
            f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=1',
            timeout=15
        )
        
        if response.status_code == 429:
            st.session_state.bot_state['last_api_call'] = time.time() + 60
            return pd.DataFrame()
            
        if response.status_code == 200:
            ohlc = response.json()
            if isinstance(ohlc, list) and len(ohlc) > 0:
                df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df.set_index('timestamp').sort_index()
    except Exception as e:
        pass
    
    return pd.DataFrame()

def safe_update_market_data():
    """Safe data update with user-friendly rate limiting"""
    if st.session_state.bot_state['update_in_progress']:
        return
    
    current_time = time.time()
    time_since_last = current_time - st.session_state.bot_state['last_api_call']
    
    if time_since_last < API_COOLDOWN:
        remaining = API_COOLDOWN - time_since_last
        st.error(f"API cooldown: Please wait {int(remaining)} seconds")
        return
    
    st.session_state.bot_state['update_in_progress'] = True
    
    try:
        with st.spinner("Updating data (1+ minute required)..."):
            for pair in CRYPTO_PAIRS:
                data = fetch_market_data(pair)
                if not data.empty:
                    st.session_state.bot_state['historical_data'][pair] = data
                time.sleep(API_COOLDOWN)
    finally:
        st.session_state.bot_state['update_in_progress'] = False
        st.session_state.bot_state['last_api_call'] = time.time()

# Rest of the functions remain the same but with improved error handling

def main():
    st.set_page_config(page_title="Free Crypto Trading Bot", layout="wide")
    
    # Initialize columns first
    col1, col2 = st.columns([1, 3])
    
    # Sidebar controls
    with st.sidebar:
        st.header("Trading Controls")
        selected_pair = st.selectbox("Asset Pair", CRYPTO_PAIRS)
        selected_strategy = st.selectbox("Trading Strategy", STRATEGIES)
        
        update_disabled = (
            st.session_state.bot_state['update_in_progress'] or 
            (time.time() - st.session_state.bot_state['last_api_call'] < API_COOLDOWN)
        )
        
        if st.button("🔄 Update Market Data", disabled=update_disabled):
            safe_update_market_data()
        
        # Show cooldown status
        elapsed = time.time() - st.session_state.bot_state['last_api_call']
        if elapsed < API_COOLDOWN:
            remaining = API_COOLDOWN - elapsed
            st.warning(f"Next update available in {int(remaining)} seconds")

    with col1:
        st.metric("Available Capital", f"${st.session_state.bot_state['capital']:,.2f}")
        st.metric("Open Positions", len(st.session_state.bot_state['positions']))
        
        if st.session_state.bot_state['positions']:
            st.write("### Current Positions")
            for pair, position in st.session_state.bot_state['positions'].items():
                current_price = position['entry_price']
                try:
                    if pair in st.session_state.bot_state['historical_data']:
                        current_price = st.session_state.bot_state['historical_data'][pair]['close'].iloc[-1]
                except:
                    pass
                
                pnl = (current_price / position['entry_price'] - 1) * 100
                st.write(f"""
                **{pair}**  
                Quantity: {position['quantity']:.4f}  
                Entry: ${position['entry_price']:.2f}  
                Current: ${current_price:.2f} ({pnl:.2f}%)  
                SL: ${position['stop_loss']:.2f}  
                TP: ${position['take_profit']:.2f}
                """)

    with col2:
        st.header("Market Analysis")
        
        if selected_pair in st.session_state.bot_state['historical_data']:
            data = st.session_state.bot_state['historical_data'][selected_pair]
            data = calculate_technical_indicators(data)
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index,
                                        open=data['open'],
                                        high=data['high'],
                                        low=data['low'],
                                        close=data['close'],
                                        name='Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50'))
            fig.update_layout(title=f"{selected_pair} Technical Analysis",
                            xaxis_title="Time",
                            yaxis_title="Price (USD)",
                            template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            final_value = run_backtest(data)
            st.metric("Strategy Backtest Result", 
                     f"${final_value:,.2f}", 
                     f"{(final_value/st.session_state.bot_state['capital']-1)*100:.2f}%")
        else:
            st.info("Data not available - Please update market data first")

if __name__ == "__main__":
    main()
