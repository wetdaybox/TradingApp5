# app.py (with explicit trade rules)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Trading System", layout="wide")
st.title("📈 Free Trading System with Rules")

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
    
    # Calculate trade parameters when signal changes
    df['Entry_Price'] = np.where(df['Signal'].diff() == 1, df['Price'], np.nan)
    df['Stop_Loss'] = df['Entry_Price'] * (1 - risk_pct)
    df['Take_Profit'] = df['Entry_Price'] * (1 + (risk_pct * reward_ratio))
    
    return df.fillna(method='ffill')

with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Stock", "AAPL").upper()
    years = st.slider("Years History", 1, 5, 3)
    risk = st.slider("Risk %", 1.0, 10.0, 5.0) / 100
    reward = st.selectbox("Reward Ratio", [2, 3, 4], index=0)

df = get_data(ticker, days=years*365)
if not df.empty:
    df = calculate_strategy(df, risk_pct=risk, reward_ratio=reward)
    
    # Current trade status
    current_signal = df['Signal'].iloc[-1]
    last_trade = df[df['Entry_Price'].notna()].iloc[-1] if current_signal else None
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${df['Price'].iloc[-1]:.2f}")
    
    if current_signal:
        col2.metric("Entry Price", f"${last_trade['Entry_Price']:.2f}", 
                   delta="ACTIVE LONG")
        col3.metric("Exit Targets", 
                   f"Stop: ${last_trade['Stop_Loss']:.2f}\nProfit: ${last_trade['Take_Profit']:.2f}")
    else:
        col2.metric("Market Position", "FLAT")
        col3.metric("Next Buy Signal", 
                   "Price > 50-day SMA", 
                   delta="WAITING")
    
    # Strategy plot with annotations
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, df['Price'], label='Price', lw=1.5, color='navy')
    ax.plot(df.index, df['SMA'], label='50-day SMA', ls='--', color='orange')
    
    if current_signal:
        ax.axhline(last_trade['Stop_Loss'], color='red', ls=':', 
                  label=f'Stop Loss ({risk*100:.0f}%)')
        ax.axhline(last_trade['Take_Profit'], color='green', ls=':', 
                  label=f'Take Profit ({risk*100*reward:.0f}%)')
        ax.fill_between(df.index, df['Stop_Loss'], df['Take_Profit'], 
                       where=df['Signal']==1, color='lightgreen', alpha=0.2)
    
    ax.set_title(f"{ticker} Trading Plan")
    ax.legend()
    st.pyplot(fig)
    
    # Trade rules explanation
    with st.expander("Trading Rules"):
        st.write(f"""
        **Entry Signal:**
        - Buy when price closes above 50-day SMA
        - Size position using 5% account risk
        
        **Exit Rules:**
        - Stop Loss: {risk*100:.0f}% below entry
        - Take Profit: {risk*100*reward:.0f}% above entry
        - Close position if price crosses back below SMA
        
        **Money Management:**
        - Risk/Reward Ratio: 1:{reward}
        - Position size calculated using:
        `Shares = (Account Risk) / (Entry Price × Risk %)`
        """)

st.markdown("---\n*Educational purpose only. Not financial advice.*")
