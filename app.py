# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import websockets
import json
import asyncio
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Configuration
SYMBOL = 'BTC-USD'
TIMEFRAME = '5m'
RISK_PARAMS = {
    'stop_loss_pct': 2.0,
    'take_profit_pct': 4.0,
    'max_position': 0.1
}

async def get_realtime_price():
    """Get real-time price data from Yahoo Finance"""
    ticker = yf.Ticker(SYMBOL)
    while True:
        try:
            data = ticker.history(period='1d', interval='1m')
            yield data.iloc[-1]
            await asyncio.sleep(30)  # Update every 30 seconds
        except Exception as e:
            st.error(f"Data error: {str(e)}")
            await asyncio.sleep(10)

def calculate_indicators(df):
    """Calculate technical indicators"""
    df['rsi'] = RSIIndicator(df['Close']).rsi()
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    return df

def generate_signal(data):
    """Generate trading signals"""
    signal = {
        'timestamp': datetime.now(),
        'price': data.Close,
        'action': 'HOLD'
    }
    
    if data.Close < data.bb_lower and data.rsi < 35:
        signal.update({
            'action': 'BUY',
            'stop_loss': data.Close * (1 - RISK_PARAMS['stop_loss_pct']/100),
            'take_profit': data.Close * (1 + RISK_PARAMS['take_profit_pct']/100)
        })
    elif data.Close > data.bb_upper and data.rsi > 70:
        signal['action'] = 'SELL'
        
    return signal

async def main():
    """Main trading logic"""
    st.title("💰 Real-Time Crypto Trading Signals")
    
    # Initialize display elements
    price_placeholder = st.empty()
    chart_placeholder = st.empty()
    signal_placeholder = st.empty()
    
    # Historical data initialization
    hist_data = yf.download(SYMBOL, period='5d', interval=TIMEFRAME)
    hist_data = calculate_indicators(hist_data)
    
    async for live_data in get_realtime_price():
        # Update historical data
        hist_data = pd.concat([hist_data, live_data.to_frame().T])
        hist_data = hist_data.iloc[-100:]
        hist_data = calculate_indicators(hist_data)
        
        # Generate signal
        signal = generate_signal(hist_data.iloc[-1])
        
        # Update displays
        price_placeholder.metric(
            "Current Price", 
            f"${hist_data.iloc[-1].Close:.2f}",
            f"{hist_data.iloc[-1].Close - hist_data.iloc[-2].Close:.2f}"
        )
        
        # Update chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist_data.index,
            open=hist_data.Open,
            high=hist_data.High,
            low=hist_data.Low,
            close=hist_data.Close
        ))
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data.bb_upper,
            line=dict(color='red'),
            name='Upper Band'
        ))
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data.bb_lower,
            line=dict(color='green'),
            name='Lower Band'
        ))
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Show signals
        if signal['action'] != 'HOLD':
            signal_placeholder.success(f"""
            🚨 **Trading Signal** ({signal['timestamp'].strftime('%H:%M:%S')})
            - Action: {signal['action']}
            - Price: ${signal['price']:.2f}
            - Stop Loss: ${signal.get('stop_loss', 'N/A'):.2f}
            - Take Profit: ${signal.get('take_profit', 'N/A'):.2f}
            """)
        else:
            signal_placeholder.info("Monitoring market - no signals detected")

if __name__ == "__main__":
    asyncio.run(main())
