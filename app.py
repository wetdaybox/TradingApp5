# app.py (mobile-optimized)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Mobile-friendly configuration
st.set_page_config(page_title="Trading System", layout="wide")
plt.style.use('ggplot')  # Better contrast for mobile screens

@st.cache_data
def get_data(ticker='AAPL', days=1095):
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[['Close']].rename(columns={'Close':'Price'})
    df.index = pd.to_datetime(df.index)
    df = df.resample('B').last().ffill()
    return df.dropna()

def calculate_strategy(df, sma_window=50, risk_pct=0.05, reward_ratio=2):
    df['SMA'] = df['Price'].rolling(sma_window).mean()
    df['Signal'] = (df['Price'] > df['SMA']).astype(int).shift(1).fillna(0)
    
    df['Entry_Price'] = np.where(df['Signal'].diff() == 1, df['Price'], np.nan)
    df['Stop_Loss'] = df['Entry_Price'] * (1 - risk_pct)
    df['Take_Profit'] = df['Entry_Price'] * (1 + (risk_pct * reward_ratio))
    
    return df.ffill()

with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Stock", "AAPL").upper()
    years = st.slider("Years History", 1, 5, 3)
    risk = st.slider("Risk %", 1.0, 10.0, 5.0) / 100
    reward = st.selectbox("Reward Ratio", [2, 3, 4], index=0)

df = get_data(ticker, days=years*365)
if not df.empty:
    df = calculate_strategy(df, risk_pct=risk, reward_ratio=reward)
    
    current_signal = df['Signal'].iloc[-1]
    last_trade = df[df['Entry_Price'].notna()].iloc[-1] if current_signal else None
    
    # Mobile-optimized layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Status")
        st.metric("Price", f"${df['Price'].iloc[-1]:.2f}")
        
    with col2:
        if current_signal:
            st.subheader("Active Trade")
            st.metric("Entry Price", f"${last_trade['Entry_Price']:.2f}")
            st.metric("Stop Loss", f"${last_trade['Stop_Loss']:.2f}", 
                     delta=f"-{risk*100:.0f}%")
            st.metric("Take Profit", f"${last_trade['Take_Profit']:.2f}", 
                     delta=f"+{risk*100*reward:.0f}%")
        else:
            st.subheader("Market Position")
            st.write("No active positions")
            st.metric("Next Signal", 
                     "Price > 50-day SMA", 
                     delta="Waiting for entry")
    
    # Enhanced plot with annotations
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Price'], label='Price', lw=2, color='#2c7bb6')
    ax.plot(df.index, df['SMA'], label='50-day SMA', ls='--', color='#d7191c')
    
    if current_signal:
        ax.axhline(last_trade['Stop_Loss'], color='#ff0000', lw=2, ls='-',
                  label=f'Stop Loss ({last_trade["Stop_Loss"]:.2f})')
        ax.axhline(last_trade['Take_Profit'], color='#008000', lw=2, ls='-',
                  label=f'Take Profit ({last_trade["Take_Profit"]:.2f})')
        
        # Add text annotations
        ax.text(df.index[-1], last_trade['Stop_Loss'], 
               f' STOP\n${last_trade["Stop_Loss"]:.2f}',
               va='center', ha='right', color='#ff0000')
        ax.text(df.index[-1], last_trade['Take_Profit'], 
               f' TAKE PROFIT\n${last_trade["Take_Profit"]:.2f}',
               va='center', ha='right', color='#008000')
    
    ax.set_title(f"{ticker} Trading Plan", fontsize=14, weight='bold')
    ax.legend(loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

# Mobile-optimized expander
with st.expander("Trading Rules 📖", expanded=True):
    st.write(f"""
    **Entry Signal** 📈
    - Buy when closing price > 50-day SMA
    - Size position using {risk*100:.0f}% account risk
    
    **Exit Rules** 🚪
    - Stop Loss: {risk*100:.0f}% below entry price
    - Take Profit: {risk*100*reward:.0f}% above entry price
    
    **Risk Management** ⚖️
    - Risk/Reward Ratio: 1:{reward}
    - Position size: (Account Risk) / (Entry Price × Risk %)
    """)

st.markdown("---\n*Mobile-optimized trading system - Refresh for latest data*")
