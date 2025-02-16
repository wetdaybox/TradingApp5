# app.py (fixed version)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure the app
st.set_page_config(page_title="Free Trading System", layout="wide")
st.title("📈 Free Adaptive Trading System")
st.markdown("""
A completely free trading system using public market data. Runs entirely in your browser via Streamlit Cloud.
""")

# Cache data processing for better performance
@st.cache_data
def get_data(days=365*3, ticker='AAPL'):
    """Get and preprocess data with built-in error handling"""
    end = datetime.today()
    start = end - timedelta(days=days)
    
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            st.error("No data found - try a different ticker")
            return pd.DataFrame()
            
        # Create uniform business day index
        df = df[['Close']].rename(columns={'Close':'Price'})
        df.index = pd.to_datetime(df.index)
        all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        df = df.reindex(all_days).ffill()
        
        return df.dropna()
    
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def calculate_strategy(df, leverage=3, sma_window=50):
    """Core trading logic - FIXED VERSION"""
    df['SMA'] = df['Price'].rolling(sma_window).mean()
    # Fixed line: Convert to pandas Series before using shift
    df['Signal'] = (df['Price'] > df['SMA']).astype(int).shift(1).fillna(0)
    df['Returns'] = df['Price'].pct_change()
    df['Strategy'] = (df['Signal'] * df['Returns'] * leverage).cumsum().apply(np.exp)
    return df

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Stock Ticker", "AAPL").upper()
    years = st.slider("Years History", 1, 10, 3)
    leverage = st.select_slider("Leverage", options=[1, 2, 3, 4, 5], value=3)
    stop_loss = st.slider("Stop Loss (%)", 1.0, 20.0, 5.0) / 100

# Main display
df = get_data(days=years*365, ticker=ticker)

if not df.empty:
    df = calculate_strategy(df, leverage=leverage)
    
    # Current status
    current_price = df['Price'].iloc[-1]
    current_sma = df['SMA'].iloc[-1]
    position = "ABOVE" if current_price > current_sma else "BELOW"
    
    # Trading signals
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("50-Day SMA", f"${current_sma:.2f}", delta=position)
    col3.metric("Strategy Return", 
               f"{(df['Strategy'].iloc[-1] - 1)*100:.1f}%", 
               "LONG" if df['Signal'].iloc[-1] else "FLAT")
    
    # Strategy plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Price'], label='Price', color='navy')
    ax.plot(df.index, df['SMA'], label='50-day SMA', linestyle='--', color='orange')
    ax.set_title(f"{ticker} Trading Strategy Performance")
    ax.legend()
    st.pyplot(fig)
    
    # Risk management calculator
    with st.expander("Position Size Calculator"):
        capital = st.number_input("Account Balance", 1000, 1000000, 10000)
        risk_amount = capital * (stop_loss)
        shares = int(risk_amount / (current_price * stop_loss))
        st.write(f"""
        - **Risk Per Trade:** ${risk_amount:.2f}
        - **Shares to Buy:** {shares} ({shares*current_price:.2f} value)
        - **Stop Price:** ${current_price*(1-stop_loss):.2f}
        """)

st.markdown("---\n*Educational use only. Past performance ≠ future results.*")
