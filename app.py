import streamlit as st
import pandas as pd
import numpy as np
import websockets
import json
import asyncio
import talib
import plotly.graph_objs as go
from datetime import datetime
from binance.client import Client

# Free API Configuration
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@kline_5m"
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"

# Trading Strategy Configuration
RISK_PARAMS = {
    'stop_loss_pct': 2.0,  # 2% stop loss
    'take_profit_pct': 4.0, # 4% take profit
    'max_position_size': 0.1 # 10% of portfolio per trade
}

class RealTimeTradingBot:
    def __init__(self):
        self.df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.portfolio = 10000  # Starting balance
        self.position = None
        self.signals = []
        
    async def connect_websocket(self):
        async with websockets.connect(BINANCE_WS_URL) as ws:
            while True:
                try:
                    data = await ws.recv()
                    await self.process_data(json.loads(data))
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
                    break

    async def process_data(self, msg):
        candle = msg['k']
        new_row = {
            'timestamp': datetime.fromtimestamp(candle['t']/1000),
            'open': float(candle['o']),
            'high': float(candle['h']),
            'low': float(candle['l']),
            'close': float(candle['c']),
            'volume': float(candle['v'])
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.analyze_market()

    def analyze_market(self):
        # Technical Indicators
        self.df['rsi'] = talib.RSI(self.df['close'], timeperiod=14)
        self.df['upper_band'], self.df['middle_band'], self.df['lower_band'] = talib.BBANDS(
            self.df['close'], timeperiod=20)
            
        current_price = self.df.iloc[-1]['close']
        rsi = self.df.iloc[-1]['rsi']
        
        # Generate Signals
        if current_price <= self.df.iloc[-1]['lower_band'] and rsi < 30:
            self.generate_signal('BUY', current_price)
        elif current_price >= self.df.iloc[-1]['upper_band'] and rsi > 70:
            self.generate_signal('SELL', current_price)

    def generate_signal(self, action, price):
        signal = {
            'timestamp': datetime.now(),
            'action': action,
            'price': price,
            'stop_loss': price * (1 - RISK_PARAMS['stop_loss_pct']/100),
            'take_profit': price * (1 + RISK_PARAMS['take_profit_pct']/100),
            'size': self.portfolio * RISK_PARAMS['max_position_size']
        }
        self.signals.append(signal)
        self.execute_trade(signal)

    def execute_trade(self, signal):
        # Simulated trading (replace with real API calls)
        if signal['action'] == 'BUY':
            self.position = signal
            st.success(f"📈 BUY Signal Executed: {signal}")
        else:
            if self.position:
                profit = (signal['price'] - self.position['price']) * self.position['size']
                self.portfolio += profit
                self.position = None
                st.success(f"📉 SELL Signal Executed: Profit ${profit:.2f}")

# Streamlit Interface
def main():
    st.title("💰 Free Crypto Trading Bot")
    
    bot = RealTimeTradingBot()
    
    # Live Data Display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${bot.df['close'].iloc[-1] if not bot.df.empty else 'Loading...'}")
    with col2:
        st.metric("Portfolio Value", f"${bot.portfolio:.2f}")
    with col3:
        st.metric("Open Positions", "Active" if bot.position else "None")
    
    # Price Chart
    fig = go.Figure()
    if not bot.df.empty:
        fig.add_trace(go.Candlestick(x=bot.df['timestamp'],
                        open=bot.df['open'],
                        high=bot.df['high'],
                        low=bot.df['low'],
                        close=bot.df['close'],
                        name='Market Data'))
        fig.add_trace(go.Scatter(x=bot.df['timestamp'], 
                             y=bot.df['upper_band'],
                             line=dict(color='red'),
                             name='Upper Band'))
        fig.add_trace(go.Scatter(x=bot.df['timestamp'],
                             y=bot.df['lower_band'],
                             line=dict(color='green'),
                             name='Lower Band'))
    st.plotly_chart(fig)
    
    # Trading Signals
    st.subheader("🚦 Live Trading Signals")
    if bot.signals:
        latest_signal = bot.signals[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Action", latest_signal['action'])
        col2.metric("Entry Price", f"${latest_signal['price']:.2f}")
        col3.metric("Position Size", f"${latest_signal['size']:.2f}")
        st.write(f"Stop Loss: ${latest_signal['stop_loss']:.2f}")
        st.write(f"Take Profit: ${latest_signal['take_profit']:.2f}")
    
    # Start WebSocket Connection
    asyncio.run(bot.connect_websocket())

if __name__ == "__main__":
    main()
