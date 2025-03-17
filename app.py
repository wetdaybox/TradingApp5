import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
REFRESH_INTERVAL = 60  # Seconds

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")

def safe_yfinance_fetch(pair):
    """Robust data fetching with error handling"""
    try:
        data = yf.download(pair, period='2d', interval='5m', progress=False)
        if not data.empty:
            data['RSI'] = calculate_rsi(data)
            st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        return data
    except Exception as e:
        st.error(f"Data Error: {str(e)}")
        return pd.DataFrame()

def calculate_rsi(data, window=14):
    """Failsafe RSI calculation"""
    try:
        prices = data['Close'] if 'Close' in data else data['Adj Close']
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)  # Prevent division by zero
        return 100 - (100 / (1 + rs))
    except Exception as e:
        st.error(f"RSI Error: {str(e)}")
        return pd.Series([50]*len(data))

def main():
    st.set_page_config(page_title="Crypto Trader Pro", layout="wide")
    st.title("📈 Real-Time Crypto Assistant")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Asset:", CRYPTO_PAIRS)
        use_manual = st.checkbox("Enter Price Manually")
        if use_manual:
            manual_price = st.number_input("Manual Price (£)", min_value=0.01, value=1000.0)
        else:
            manual_price = None
            
    with col2:
        st.caption(f"Last update: {st.session_state.last_update} UTC")
        
        # Data loading with status
        data_load_status = st.empty()
        data_load_status.info("🔄 Loading market data...")
        
        try:
            st.session_state.data = safe_yfinance_fetch(pair)
            if st.session_state.data.empty:
                data_load_status.error("❌ Failed to load data")
                return
                
            data_load_status.success(f"✅ Data loaded for {pair}")
            
            # Price display
            current_price = get_current_price(st.session_state.data, manual_price)
            if not current_price:
                st.warning("⚠️ Price data unavailable")
                return
                
            # Display trading interface
            display_trading_interface(current_price)
            
        except Exception as e:
            st.error(f"Critical Error: {str(e)}")

def get_current_price(data, manual_price):
    """Get validated current price"""
    if manual_price:
        return manual_price
        
    try:
        prices = data['Close'] if 'Close' in data else data['Adj Close']
        fx_rate = yf.download(FX_PAIR, period='1d', progress=False)['Close'].iloc[-1]
        return prices.iloc[-1] / fx_rate
    except:
        return None

def display_trading_interface(current_price):
    """Core trading display"""
    # Dynamic calculations
    stop_loss = current_price * 0.95
    take_profit = current_price * 1.15
    rsi = st.session_state.data['RSI'].iloc[-1] if 'RSI' in st.session_state.data else 50
    
    # Strategy logic
    recommendation = "Buy" if rsi < RSI_OVERSOLD else "Sell" if rsi > RSI_OVERBOUGHT else "Hold"
    
    # Display metrics
    cols = st.columns(3)
    cols[0].metric("Current Price", f"£{current_price:.4f}")
    cols[1].metric("24h Range", f"£{st.session_state.data['Low'].min():.4f}-£{st.session_state.data['High'].max():.4f}")
    cols[2].metric("RSI", f"{rsi:.1f}", delta="Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral")
    
    # Trading strategy
    with st.expander("Trading Strategy", expanded=True):
        st.write(f"""
        **Recommended Action:** {recommendation}
        **Take Profit:** £{take_profit:.4f} (+15%)
        **Stop Loss:** £{stop_loss:.4f} (-5%)
        """)
    
    # Price chart
    plot_price_chart()

def plot_price_chart():
    """Failsafe chart plotting"""
    try:
        fig = go.Figure(data=[go.Candlestick(
            x=st.session_state.data.index,
            open=st.session_state.data['Open'],
            high=st.session_state.data['High'],
            low=st.session_state.data['Low'],
            close=st.session_state.data['Close'] if 'Close' in st.session_state.data else st.session_state.data['Adj Close']
        )])
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart Error: {str(e)}")

if __name__ == "__main__":
    main()
