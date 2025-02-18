import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import backtrader as bt
import requests
import pytz
from datetime import datetime, timedelta
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import plotly.graph_objects as go

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'XRP-USD']
STRATEGIES = ['SMA Crossover', 'RSI Divergence', 'MACD Momentum']
RISK_PARAMS = {
    'max_risk_per_trade': 0.02,  # 2% of capital
    'stop_loss_pct': 0.05,       # 5%
    'take_profit_pct': 0.10      # 10%
}

# Initialize session state
if 'bot_state' not in st.session_state:
    st.session_state.bot_state = {
        'positions': {},
        'capital': 10000,
        'historical_data': {},
        'performance': pd.DataFrame(),
        'last_update': datetime.now(pytz.utc)
    }

class AdvancedStrategy(bt.Strategy):
    params = (
        ('sma_fast', 20),
        ('sma_slow', 50),
        ('rsi_period', 14),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9)
    )

    def __init__(self):
        self.sma_fast = bt.indicators.SMA(period=self.p.sma_fast)
        self.sma_slow = bt.indicators.SMA(period=self.p.sma_slow)
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
        self.macd = bt.indicators.MACD(
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )

    def next(self):
        if not self.position:
            if self.sma_fast > self.sma_slow and self.rsi < 30:
                size = self.broker.getvalue() * RISK_PARAMS['max_risk_per_trade'] / self.data.close[0]
                self.buy(size=size)
        else:
            if self.sma_fast < self.sma_slow and self.rsi > 70:
                self.sell()

def fetch_market_data(pair):
    """Multi-source data fetching with fallback"""
    try:
        data = yf.download(pair, period='1d', interval='5m')
        if not data.empty:
            return data
    except:
        pass
    
    try:
        response = requests.get(f'https://api.coingecko.com/api/v3/coins/{pair.split("-")[0].lower()}/ohlc?vs_currency=usd&days=1')
        ohlc = [x[1:] for x in response.json()]
        df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')
    except:
        return pd.DataFrame()

def calculate_technical_indicators(data):
    """Advanced technical analysis using ta library"""
    # RSI
    rsi_indicator = RSIIndicator(data['close'], window=14)
    data['RSI'] = rsi_indicator.rsi()
    
    # MACD
    macd_indicator = MACD(data['close'], window_fast=12, window_slow=26, window_sign=9)
    data['MACD'] = macd_indicator.macd()
    data['MACD_Signal'] = macd_indicator.macd_signal()
    
    # Bollinger Bands
    bb_indicator = BollingerBands(data['close'], window=20, window_dev=2)
    data['Bollinger_Upper'] = bb_indicator.bollinger_hband()
    data['Bollinger_Lower'] = bb_indicator.bollinger_lband()
    
    # SMA
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    
    return data.dropna()

def execute_paper_trade(pair, action, price, quantity):
    """Paper trading engine"""
    if action == 'BUY':
        cost = price * quantity
        if cost > st.session_state.bot_state['capital']:
            return False
        st.session_state.bot_state['capital'] -= cost
        st.session_state.bot_state['positions'][pair] = {
            'quantity': quantity,
            'entry_price': price,
            'stop_loss': price * (1 - RISK_PARAMS['stop_loss_pct']),
            'take_profit': price * (1 + RISK_PARAMS['take_profit_pct'])
        }
        return True
    elif action == 'SELL':
        position = st.session_state.bot_state['positions'].get(pair)
        if position:
            proceeds = price * position['quantity']
            st.session_state.bot_state['capital'] += proceeds
            del st.session_state.bot_state['positions'][pair]
            return True
    return False

def run_backtest(data):
    """Backtesting engine"""
    cerebro = bt.Cerebro()
    cerebro.addstrategy(AdvancedStrategy)
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(st.session_state.bot_state['capital'])
    cerebro.broker.setcommission(commission=0.001)  # 0.1% trading fee
    
    results = cerebro.run()
    return cerebro.broker.getvalue()

def main():
    st.set_page_config(page_title="Advanced Crypto Trading Bot", layout="wide")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Trading Controls")
        selected_pair = st.selectbox("Asset Pair", CRYPTO_PAIRS)
        selected_strategy = st.selectbox("Trading Strategy", STRATEGIES)
        risk_level = st.slider("Risk Level (%)", 1, 10, 2)
        
        if st.button("🔄 Update Market Data"):
            update_market_data()
    
    # Main interface
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("Available Capital", f"${st.session_state.bot_state['capital']:,.2f}")
        st.metric("Open Positions", len(st.session_state.bot_state['positions']))
        
        if st.session_state.bot_state['positions']:
            st.write("### Current Positions")
            for pair, position in st.session_state.bot_state['positions'].items():
                current_price = yf.Ticker(pair).history(period='1d').iloc[-1]['Close']
                pnl = (current_price / position['entry_price'] - 1) * 100
                st.write(f"""
                **{pair}**  
                Quantity: {position['quantity']:.4f}  
                Entry: ${position['entry_price']:.2f}  
                Current: ${current_price:.2f} ({pnl:.2f}%)  
                SL: ${position['stop_loss']:.2f}  
                TP: ${position['take_profit']:.2f}
                """)
    
    with col2:
        st.header("Market Analysis")
        data = fetch_market_data(selected_pair)
        if not data.empty:
            data = calculate_technical_indicators(data)
            
            # Display price chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index,
                                        open=data['open'],
                                        high=data['high'],
                                        low=data['low'],
                                        close=data['close'],
                                        name='Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50'))
            fig.update_layout(title=f"{selected_pair} Technical Analysis",
                            xaxis_title="Time",
                            yaxis_title="Price (USD)",
                            template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # Run backtest
            final_value = run_backtest(data)
            st.metric("Strategy Backtest Result", 
                     f"${final_value:,.2f}", 
                     f"{(final_value/st.session_state.bot_state['capital']-1)*100:.2f}%")
        else:
            st.error("Failed to load market data")

if __name__ == "__main__":
    main()
