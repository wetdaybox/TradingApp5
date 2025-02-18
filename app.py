import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import backtrader as bt
import requests
import pytz
import time
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
API_COOLDOWN = 65  # Seconds between API calls

def create_initial_data():
    """Create realistic sample data for initial training"""
    base_prices = {'BTC-USD': 45000, 'ETH-USD': 2500, 'XRP-USD': 0.55}
    sample_data = {}
    
    for pair in CRYPTO_PAIRS:
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=288, freq='5min')
        base_price = base_prices[pair]
        volatility = 0.015 + (0.005 if pair == 'XRP-USD' else 0)
        
        close = base_price * (1 + np.cumsum(volatility * np.random.randn(288)) / 100
        open = close * 0.998
        high = close * 1.005
        low = close * 0.995
        
        sample_data[pair] = pd.DataFrame({
            'Open': open,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': np.random.randint(1000, 5000, 288)
        }, index=dates)
    
    return sample_data

# Initialize session state with robust sample data
if 'bot_state' not in st.session_state:
    st.session_state.update({
        'bot_state': {
            'positions': {},
            'capital': 10000,
            'historical_data': create_initial_data(),
            'performance': pd.DataFrame(),
            'last_update': datetime.now(pytz.utc),
            'last_api_call': 0,
            'update_in_progress': False,
            'using_sample_data': True
        }
    })

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_market_data(pair):
    """Multi-source data fetching with strict rate limiting"""
    current_time = time.time()
    
    # Enforce API cooldown
    if current_time - st.session_state.bot_state['last_api_call'] < API_COOLDOWN:
        return pd.DataFrame()
    
    try:
        data = yf.download(pair, period='1d', interval='5m', progress=False)
        if not data.empty:
            st.session_state.bot_state['last_api_call'] = time.time()
            return data
    except Exception as e:
        st.warning(f"Yahoo Finance failed: {str(e)}")
    
    try:
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
        st.error(f"Data fetch failed: {str(e)}")
    
    return pd.DataFrame()

def safe_update_market_data():
    """Data update with user-friendly rate limiting"""
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
        with st.spinner("Fetching real market data..."):
            new_data = {}
            for pair in CRYPTO_PAIRS:
                data = fetch_market_data(pair)
                if not data.empty:
                    new_data[pair] = data
                time.sleep(API_COOLDOWN)
            
            if new_data:
                st.session_state.bot_state['historical_data'] = new_data
                st.session_state.bot_state['using_sample_data'] = False
                st.success("Real data loaded successfully!")
    finally:
        st.session_state.bot_state['update_in_progress'] = False
        st.session_state.bot_state['last_api_call'] = time.time()

def calculate_technical_indicators(data):
    """Technical analysis with validation"""
    if data.empty:
        return data
    
    try:
        data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
        macd = MACD(data['Close'], window_fast=12, window_slow=26, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        bb = BollingerBands(data['Close'], window=20, window_dev=2)
        data['Bollinger_Upper'] = bb.bollinger_hband()
        data['Bollinger_Lower'] = bb.bollinger_lband()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        return data.dropna()
    except Exception as e:
        st.error(f"Indicator calculation failed: {str(e)}")
        return data

def main():
    st.set_page_config(page_title="Crypto Trading Bot", layout="wide")
    
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
        
        if st.session_state.bot_state['using_sample_data']:
            st.warning("Using initial sample data - update to get real market data")
        else:
            st.success("Using real market data")
            
        elapsed = time.time() - st.session_state.bot_state['last_api_call']
        if elapsed < API_COOLDOWN:
            remaining = API_COOLDOWN - elapsed
            st.info(f"Next update available in {int(remaining)} seconds")

    # Positions panel
    with col1:
        st.metric("Available Capital", f"${st.session_state.bot_state['capital']:,.2f}")
        st.metric("Open Positions", len(st.session_state.bot_state['positions']))
        
        if st.session_state.bot_state['positions']:
            st.write("### Current Positions")
            for pair, position in st.session_state.bot_state['positions'].items():
                try:
                    data = st.session_state.bot_state['historical_data'].get(pair)
                    current_price = data['Close'].iloc[-1] if data is not None else position['entry_price']
                except:
                    current_price = position['entry_price']
                
                pnl = (current_price / position['entry_price'] - 1) * 100
                st.write(f"""
                **{pair}**  
                Quantity: {position['quantity']:.4f}  
                Entry: ${position['entry_price']:.2f}  
                Current: ${current_price:.2f} ({pnl:.2f}%)  
                SL: ${position['stop_loss']:.2f}  
                TP: ${position['take_profit']:.2f}
                """)

    # Market analysis
    with col2:
        st.header("Market Analysis")
        
        data = st.session_state.bot_state['historical_data'].get(selected_pair)
        if data is not None and not data.empty:
            data = calculate_technical_indicators(data)
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50'))
            fig.update_layout(
                title=f"{selected_pair} Technical Analysis",
                xaxis_title="Time",
                yaxis_title="Price (USD)",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            try:
                final_value = run_backtest(data)
                st.metric("Strategy Backtest Result", 
                         f"${final_value:,.2f}", 
                         f"{(final_value/st.session_state.bot_state['capital']-1)*100:.2f}%")
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
        else:
            st.info("No market data available - use the update button to load data")

if __name__ == "__main__":
    main()
