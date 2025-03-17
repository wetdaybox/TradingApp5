import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
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
    st.session_state.last_update = "Initializing..."

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
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / (avg_loss + 1e-10)  # Prevent division by zero
        return 100 - (100 / (1 + rs))
    except Exception as e:
        st.error(f"RSI Error: {str(e)}")
        return pd.Series([50]*len(data))  # Neutral default

def main():
    st.set_page_config(page_title="Crypto Trader Pro", layout="wide")
    st.title("📈 Real-Time Crypto Assistant")
    
    with st.spinner('Loading trading engine...'):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            pair = st.selectbox("Select Asset:", CRYPTO_PAIRS)
            use_manual = st.checkbox("Enter Price Manually")
            manual_price = st.number_input("Manual Price (£)", 
                                         min_value=0.01, 
                                         value=1000.0) if use_manual else None
            
        with col2:
            st.caption(f"Last update: {st.session_state.last_update} UTC")
            
            try:
                # Load and validate data
                st.session_state.data = safe_yfinance_fetch(pair)
                if st.session_state.data.empty:
                    st.warning("⚠️ Market data unavailable - try refreshing")
                    return
                
                # Get current price
                current_price = get_current_price(manual_price)
                if current_price is None:
                    st.warning("⚠️ Price data unavailable")
                    return
                
                # Display interface
                display_trading_interface(current_price)
                st_autorefresh(interval=REFRESH_INTERVAL*1000, key="refresh")
                
            except Exception as e:
                st.error(f"System Error: {str(e)}")

def get_current_price(manual_price):
    """Get validated current price"""
    if manual_price:
        return manual_price
        
    try:
        data = st.session_state.data
        prices = data['Close'] if 'Close' in data else data['Adj Close']
        fx_data = yf.download(FX_PAIR, period='1d', progress=False)
        fx_rate = fx_data['Close'].iloc[-1] if not fx_data.empty else 0.80
        return prices.iloc[-1] / fx_rate
    except Exception as e:
        st.error(f"Price Error: {str(e)}")
        return None

def display_trading_interface(current_price):
    """Core trading display with error handling"""
    try:
        data = st.session_state.data
        
        # Get RSI value safely
        rsi_value = data['RSI'].iloc[-1].item() if 'RSI' in data.columns else 50
        
        # Get price extremes
        low = data['Low'].min().item() if 'Low' in data.columns else current_price
        high = data['High'].max().item() if 'High' in data.columns else current_price
        
        # Calculate trading parameters
        stop_loss = current_price * 0.95  # 5% stop loss
        take_profit = current_price * 1.15  # 15% target
        
        # Generate recommendation
        recommendation = "Buy" if rsi_value < RSI_OVERSOLD else \
                       "Sell" if rsi_value > RSI_OVERBOUGHT else "Hold"
        
        # Display metrics
        cols = st.columns(3)
        cols[0].metric("Current Price", f"£{current_price:.4f}")
        cols[1].metric("24h Range", f"£{low:.4f}-£{high:.4f}")
        cols[2].metric("RSI", f"{rsi_value:.1f}", 
                      delta="Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral")
        
        # Trading strategy
        with st.expander("Trading Strategy", expanded=True):
            st.write(f"""
            **Recommended Action:** {recommendation}  
            **Take Profit:** £{take_profit:.4f} (+15%)  
            **Stop Loss:** £{stop_loss:.4f} (-5%)
            """)
        
        # Price chart
        plot_price_chart()

    except Exception as e:
        st.error(f"Display Error: {str(e)}")

def plot_price_chart():
    """Failsafe price chart"""
    try:
        data = st.session_state.data
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'] if 'Close' in data else data['Adj Close']
        )])
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart Error: {str(e)}")

if __name__ == "__main__":
    main()
