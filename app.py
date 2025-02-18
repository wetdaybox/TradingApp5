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
API_COOLDOWN = 60  # Seconds between API calls

# Initialize session state
if 'bot_state' not in st.session_state:
    st.session_state.bot_state = {
        'positions': {},
        'capital': 10000,
        'historical_data': {},
        'performance': pd.DataFrame(),
        'last_update': datetime.now(pytz.utc),
        'last_api_call': 0
    }

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_market_data(pair):
    """Multi-source data fetching with strict rate limiting"""
    current_time = time.time()
    
    # Enforce API cooldown
    if current_time - st.session_state.bot_state['last_api_call'] < API_COOLDOWN:
        remaining = API_COOLDOWN - (current_time - st.session_state.bot_state['last_api_call'])
        st.warning(f"API rate limit: Please wait {int(remaining)} seconds before next request")
        return pd.DataFrame()
    
    try:
        # Try Yahoo Finance first
        data = yf.download(pair, period='1d', interval='5m', progress=False)
        if not data.empty:
            st.session_state.bot_state['last_api_call'] = time.time()
            return data
    except Exception as e:
        st.warning(f"Yahoo Finance failed: {str(e)}")
    
    try:
        # Fallback to CoinGecko (public API)
        st.session_state.bot_state['last_api_call'] = time.time()
        coin_id = pair.split("-")[0].lower()
        response = requests.get(
            f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=1',
            timeout=10
        )
        response.raise_for_status()
        
        if response.status_code == 429:
            st.error("Public API rate limit reached. Please wait 1 minute.")
            time.sleep(60)
            return pd.DataFrame()
            
        ohlc = response.json()
        if isinstance(ohlc, list) and len(ohlc) > 0:
            df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.set_index('timestamp').sort_index()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.error("Please wait at least 1 minute between data updates")
        else:
            st.error(f"API error: {str(e)}")
    except Exception as e:
        st.error(f"Data fetch failed: {str(e)}")
    
    return pd.DataFrame()

def update_market_data():
    """Data update with user-friendly rate limiting"""
    current_time = time.time()
    if current_time - st.session_state.bot_state['last_api_call'] < API_COOLDOWN:
        remaining = API_COOLDOWN - (current_time - st.session_state.bot_state['last_api_call'])
        st.error(f"Please wait {int(remaining)} seconds before updating again")
        return
    
    with st.spinner("Updating market data (public API - please be patient)..."):
        for pair in CRYPTO_PAIRS:
            data = fetch_market_data(pair)
            if not data.empty:
                st.session_state.bot_state['historical_data'][pair] = data
                st.success(f"Updated {pair} data")
            else:
                st.error(f"Failed to update {pair} data")
            time.sleep(API_COOLDOWN)  # Public API cooldown

# Rest of the functions (calculate_technical_indicators, execute_paper_trade, 
# run_backtest, AdvancedStrategy) remain identical to previous version

def main():
    st.set_page_config(page_title="Crypto Trading Bot (Free Version)", layout="wide")
    
    # Initialize columns first
    col1, col2 = st.columns([1, 3])
    
    # Sidebar controls
    with st.sidebar:
        st.header("Trading Controls")
        selected_pair = st.selectbox("Asset Pair", CRYPTO_PAIRS)
        selected_strategy = st.selectbox("Trading Strategy", STRATEGIES)
        
        st.warning("""
        **Free API Notice:**  
        - Limited to 1 request/minute  
        - Data might be delayed  
        - For better performance, run locally
        """)
        
        if st.button("🔄 Update Market Data"):
            update_market_data()

    with col1:
        st.metric("Available Capital", f"${st.session_state.bot_state['capital']:,.2f}")
        st.metric("Open Positions", len(st.session_state.bot_state['positions']))
        
        if st.session_state.bot_state['positions']:
            st.write("### Current Positions")
            for pair, position in st.session_state.bot_state['positions'].items():
                current_price = "N/A"
                try:
                    if pair in st.session_state.bot_state['historical_data']:
                        current_price = st.session_state.bot_state['historical_data'][pair]['close'].iloc[-1]
                    else:
                        current_price = position['entry_price']
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

    with col2:
        st.header("Market Analysis")
        data = fetch_market_data(selected_pair)
        if not data.empty:
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
            st.info("No data available - Please update market data first")

if __name__ == "__main__":
    main()
