import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import asyncio
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import plotly.graph_objects as go

# Configuration
SYMBOL = 'BTC-USD'
TIMEFRAME = '5m'
RISK_PARAMS = {
    'stop_loss_pct': 2.0,
    'take_profit_pct': 4.0,
    'max_position': 0.1
}

async def get_realtime_price():
    """Fetch a single row of real-time price data."""
    ticker = yf.Ticker(SYMBOL)
    while True:
        try:
            data = ticker.history(period='1d', interval='1m')
            if data.empty:
                st.warning("No live data received. Retrying in 30 seconds...")
                await asyncio.sleep(30)
                continue
            yield data.iloc[-1]
            await asyncio.sleep(30)
        except Exception as e:
            st.error(f"Data error: {str(e)}")
            await asyncio.sleep(10)

def calculate_indicators(df):
    """Compute RSI and Bollinger Bands on the given DataFrame."""
    df = df.copy()
    df['rsi'] = RSIIndicator(df['Close']).rsi()
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    return df

def generate_signal(latest):
    """Return a trading signal based on the latest row."""
    signal = {
        'timestamp': datetime.now(),
        'price': latest.Close,
        'action': 'HOLD'
    }
    if latest.Close < latest.bb_lower and latest.rsi < 35:
        signal.update({
            'action': 'BUY',
            'stop_loss': latest.Close * (1 - RISK_PARAMS['stop_loss_pct'] / 100),
            'take_profit': latest.Close * (1 + RISK_PARAMS['take_profit_pct'] / 100)
        })
    elif latest.Close > latest.bb_upper and latest.rsi > 70:
        signal['action'] = 'SELL'
    return signal

async def main():
    st.title("💰 Real-Time Crypto Trading Signals")
    
    # Placeholders for dynamic content
    price_placeholder = st.empty()
    chart_placeholder = st.empty()
    signal_placeholder = st.empty()
    
    # Download historical data for initial chart (using 5d window and 5m intervals)
    hist_data = yf.download(SYMBOL, period='5d', interval=TIMEFRAME)
    if hist_data.empty:
        st.error("Failed to download historical data for BTC-USD. Please check your network and try again.")
        return
    hist_data = calculate_indicators(hist_data)
    
    # Main loop: update with live data
    async for live in get_realtime_price():
        # Append new live row
        new_row = live.to_frame().T
        hist_data = pd.concat([hist_data, new_row])
        hist_data = hist_data.iloc[-100:]  # Keep last 100 rows for the chart
        hist_data = calculate_indicators(hist_data)
        
        # Generate signal based on the most recent data
        latest = hist_data.iloc[-1]
        signal = generate_signal(latest)
        
        # Compute price change if possible
        if len(hist_data) > 1:
            price_change = latest.Close - hist_data.iloc[-2].Close
        else:
            price_change = 0.0
        
        # Update the price metric
        price_placeholder.metric(
            "Current Price",
            f"${latest.Close:.2f}",
            f"{price_change:.2f}"
        )
        
        # Create a candlestick chart with Bollinger Bands overlay
        fig = go.Figure(data=[go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name="Price"
        )])
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['bb_upper'],
            line=dict(color='red'),
            name='Upper Band'
        ))
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['bb_lower'],
            line=dict(color='green'),
            name='Lower Band'
        ))
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Show trading signal if not HOLD
        if signal['action'] != 'HOLD':
            stop_loss = signal.get('stop_loss', None)
            take_profit = signal.get('take_profit', None)
            signal_placeholder.success(
                f"🚨 {signal['action']} signal at {signal['timestamp'].strftime('%H:%M:%S')}\n"
                f"Price: ${signal['price']:.2f}\n"
                f"Stop Loss: ${stop_loss:.2f}" if stop_loss else "N/A" +
                f", Take Profit: ${take_profit:.2f}" if take_profit else ""
            )
        else:
            signal_placeholder.info("Monitoring market – no signal detected.")

if __name__ == "__main__":
    asyncio.run(main())
