import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf

# Strategy parameters
STRATEGIES = ['SMA Crossover', 'RSI Reversion', 'Bollinger Bands']
DEFAULT_TICKER = 'BTC-USD'

def get_data(ticker, start_date, end_date):
    """Get historical price data"""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def calculate_sma_strategy(data, short_window=20, long_window=50):
    """Simple Moving Average Crossover Strategy"""
    data['SMA_20'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_50'] = data['Close'].rolling(window=long_window).mean()
    data['Signal'] = np.where(data['SMA_20'] > data['SMA_50'], 1, -1)
    return data

def calculate_rsi(data, window=14):
    """Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def plot_strategy(data, ticker, strategy):
    """Interactive price chart with strategy signals"""
    fig = go.Figure()
    
    # Price plot
    fig.add_trace(go.Candlestick(x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'))
    
    # Strategy-specific indicators
    if strategy == 'SMA Crossover':
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='20-day SMA'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='50-day SMA'))
    elif strategy == 'RSI Reversion':
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', yaxis='y2'))
        fig.update_layout(yaxis2=dict(title='RSI', overlaying='y', side='right'))
    
    fig.update_layout(
        title=f'{ticker} {strategy} Strategy',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def backtest_strategy(data):
    """Simple backtest of strategy performance"""
    data['Returns'] = data['Close'].pct_change()
    data['Strategy'] = data['Signal'].shift(1) * data['Returns']
    data['Cumulative Returns'] = (1 + data['Strategy']).cumprod()
    return data

def main():
    st.set_page_config(page_title="Trading Bot", layout="wide")
    
    st.title("📈 Free Trading Bot")
    st.write("Historical strategy simulation using cryptocurrency data")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.text_input("Enter cryptocurrency pair:", DEFAULT_TICKER)
        strategy = st.selectbox("Select trading strategy:", STRATEGIES)
        start_date = st.date_input("Start date:", datetime(2020, 1, 1))
        end_date = st.date_input("End date:", datetime.today())
        
        if strategy == 'SMA Crossover':
            short_window = st.slider("Short SMA window:", 5, 50, 20)
            long_window = st.slider("Long SMA window:", 50, 200, 50)
        
        if st.button("Run Strategy"):
            try:
                data = get_data(ticker, start_date, end_date)
                if strategy == 'SMA Crossover':
                    data = calculate_sma_strategy(data, short_window, long_window)
                elif strategy == 'RSI Reversion':
                    data = calculate_rsi(data)
                    data['Signal'] = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))
                
                data = backtest_strategy(data)
                
                with col2:
                    plot_strategy(data, ticker, strategy)
                    st.subheader("Performance Metrics")
                    total_return = data['Cumulative Returns'].iloc[-1] - 1
                    st.metric("Total Strategy Return", f"{total_return:.2%}")
                    
                    st.write("Historical Data Preview:")
                    st.dataframe(data.tail(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
