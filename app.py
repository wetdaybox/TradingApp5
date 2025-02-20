# app.py
import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import websockets
import json
from datetime import datetime
from binance import AsyncClient, BinanceSocketManager
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Configuration
SYMBOL = 'BTCUSDT'
TIMEFRAME = '5m'
RISK_PARAMS = {
    'stop_loss_pct': 2.0,
    'take_profit_pct': 4.0,
    'max_position': 0.1
}

async def get_historical_data(client):
    """Fetch historical data for technical analysis"""
    klines = await client.get_klines(symbol=SYMBOL, interval=TIMEFRAME, limit=100)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
    return df

def calculate_indicators(df):
    """Calculate technical indicators"""
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    return df

async def handle_socket(ts):
    """Process real-time websocket data"""
    async with ts as tscm:
        while True:
            msg = await tscm.recv()
            yield msg

def create_signal(df):
    """Generate trading signals"""
    latest = df.iloc[-1]
    signal = {'timestamp': datetime.now(), 'price': latest['close']}
    
    if latest['close'] < latest['bb_lower'] and latest['rsi'] < 35:
        signal.update({
            'action': 'BUY',
            'stop_loss': latest['close'] * (1 - RISK_PARAMS['stop_loss_pct']/100),
            'take_profit': latest['close'] * (1 + RISK_PARAMS['take_profit_pct']/100)
        })
    elif latest['close'] > latest['bb_upper'] and latest['rsi'] > 70:
        signal.update({
            'action': 'SELL',
            'stop_loss': None,
            'take_profit': None
        })
    else:
        signal['action'] = 'HOLD'
    
    return signal

async def main():
    """Main trading logic"""
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    ts = bm.kline_socket(SYMBOL, interval=TIMEFRAME)
    
    # Initialize dashboard
    st.title("💰 Real-Time Crypto Trading Signals")
    price_placeholder = st.empty()
    chart_placeholder = st.empty()
    signal_placeholder = st.empty()
    
    # Get historical data
    df = await get_historical_data(client)
    df = calculate_indicators(df)
    
    async for msg in handle_socket(ts):
        # Update DataFrame with new data
        new_row = {
            'timestamp': pd.to_datetime(msg['k']['t'], unit='ms'),
            'open': float(msg['k']['o']),
            'high': float(msg['k']['h']),
            'low': float(msg['k']['l']),
            'close': float(msg['k']['c']),
            'volume': float(msg['k']['v'])
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).iloc[-100:]
        df = calculate_indicators(df)
        
        # Generate signal
        signal = create_signal(df)
        
        # Update price display
        price_placeholder.metric(
            label="Current Price",
            value=f"${df.iloc[-1]['close']:.2f}",
            delta=f"{df.iloc[-1]['close'] - df.iloc[-2]['close']:.2f}"
        )
        
        # Update chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['bb_upper'],
            line=dict(color='red'),
            name='Upper Bollinger Band'
        ))
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['bb_lower'],
            line=dict(color='green'),
            name='Lower Bollinger Band'
        ))
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Display signals
        if signal['action'] != 'HOLD':
            signal_placeholder.success(f"""
            🚨 **Trading Signal** ({signal['timestamp'].strftime('%H:%M:%S')})
            - Action: {signal['action']}
            - Price: ${signal['price']:.2f}
            - Stop Loss: ${signal.get('stop_loss', 'N/A'):.2f}
            - Take Profit: ${signal.get('take_profit', 'N/A'):.2f}
            """)
        else:
            signal_placeholder.info("No significant signals - monitoring market...")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Shutting down...")
