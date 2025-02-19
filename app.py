import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import backtrader as bt
import requests
import pytz
import time
from datetime import datetime
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import plotly.graph_objects as go
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'XRP-USD']
RISK_PARAMS = {
    'max_risk_per_trade': 0.01,
    'stop_loss_pct': 0.03,
    'take_profit_pct': 0.06
}
API_COOLDOWN = 65  # Seconds between API calls

# Enhanced data generation with realistic volatility
def create_initial_data():
    base_prices = {'BTC-USD': 45000, 'ETH-USD': 2500, 'XRP-USD': 0.55}
    sample_data = {}
    
    for pair in CRYPTO_PAIRS:
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=288, freq='5min')
        base_price = base_prices[pair]
        volatility = 0.015 + (0.005 if pair == 'XRP-USD' else 0)
        
        price_changes = np.cumsum(volatility * np.random.randn(288)) / 100
        close = base_price * (1 + price_changes)
        open_price = close * 0.998
        high = close * 1.005
        low = close * 0.995
        
        sample_data[pair] = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': np.random.randint(1000, 5000, 288)
        }, index=dates)
    
    return sample_data

# Robust Backtrader Strategy
class EnhancedStrategy(bt.Strategy):
    params = (
        ('sma_fast', 20),
        ('sma_slow', 50),
        ('rsi_period', 14),
        ('atr_period', 14),
        ('risk_per_trade', 0.01)
    )

    def __init__(self):
        self.sma_fast = bt.indicators.SMA(period=self.p.sma_fast)
        self.sma_slow = bt.indicators.SMA(period=self.p.sma_slow)
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
        self.atr = bt.indicators.ATR(period=self.p.atr_period)
        self.trade_counter = 0

    def next(self):
        if self.trade_counter >= 5:  # Max 5 trades per day
            return

        if not self.position:
            if (self.sma_fast > self.sma_slow and 
                self.rsi < 35 and 
                self.data.close[0] > self.data.open[0]):
                
                risk_amount = self.broker.getvalue() * self.p.risk_per_trade
                size = risk_amount / (self.atr[0] * 2)
                self.buy(size=size)
                self.trade_counter += 1
        else:
            if (self.data.close[0] < self.position.price * 0.97 or 
                self.data.close[0] > self.position.price * 1.06):
                self.sell()

# Initialize session state
if 'bot_state' not in st.session_state:
    st.session_state.update({
        'bot_state': {
            'positions': {},
            'capital': 10000,
            'historical_data': create_initial_data(),
            'performance': pd.DataFrame(),
            'last_api_call': 0,
            'update_in_progress': False,
            'using_sample_data': True
        }
    })

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
def fetch_market_data(pair):
    """Multi-source data fetching with enhanced reliability"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'application/json'
    }
    
    try:
        # Try Yahoo Finance with retries
        data = yf.download(
            pair,
            period='1d',
            interval='5m',
            progress=False,
            headers=headers
        )
        if not data.empty:
            return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        st.error(f"Yahoo Error: {str(e)}")
    
    try:
        # Fallback to CoinGecko
        coin_map = {'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'XRP-USD': 'ripple'}
        response = requests.get(
            f'https://api.coingecko.com/api/v3/coins/{coin_map[pair]}/ohlc?vs_currency=usd&days=1',
            headers=headers,
            timeout=15
        )
        if response.status_code == 200:
            ohlc = response.json()
            df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.set_index('timestamp').rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
            })
    except Exception as e:
        st.error(f"CoinGecko Error: {str(e)}")
    
    return pd.DataFrame()

def main():
    st.set_page_config(page_title="Crypto Trading Bot", layout="wide")
    
    st.sidebar.header("Controls")
    selected_pair = st.sidebar.selectbox("Asset", CRYPTO_PAIRS)
    
    # Data update logic
    if st.sidebar.button("🔄 Update Data") and \
        (time.time() - st.session_state.bot_state['last_api_call'] > API_COOLDOWN):
        
        with st.spinner("Updating..."):
            new_data = fetch_market_data(selected_pair)
            if not new_data.empty:
                st.session_state.bot_state['historical_data'][selected_pair] = new_data
                st.session_state.bot_state['using_sample_data'] = False
                st.session_state.bot_state['last_api_call'] = time.time()
    
    # Main display
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("Capital", f"${st.session_state.bot_state['capital']:,.2f}")
        st.write("### Risk Parameters")
        st.write(f"Max Risk/Trade: {RISK_PARAMS['max_risk_per_trade']*100}%")
        st.write(f"Stop Loss: {RISK_PARAMS['stop_loss_pct']*100}%")
        st.write(f"Take Profit: {RISK_PARAMS['take_profit_pct']*100}%")
    
    with col2:
        data = st.session_state.bot_state['historical_data'].get(selected_pair)
        if data is not None and not data.empty:
            # Enhanced technical analysis
            data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
            data['MACD'] = MACD(data['Close']).macd()
            bb = BollingerBands(data['Close'])
            data['BB_Upper'] = bb.bollinger_hband()
            data['BB_Lower'] = bb.bollinger_lband()
            
            # Interactive chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close']))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], 
                                   line=dict(color='rgba(250, 0, 0, 0.3)')))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'],
                                   line=dict(color='rgba(0, 250, 0, 0.3)')))
            fig.update_layout(height=600, title=f"{selected_pair} Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Backtesting
            if st.button("Run Backtest"):
                cerebro = bt.Cerebro()
                cerebro.addstrategy(EnhancedStrategy)
                cerebro.broker.setcash(st.session_state.bot_state['capital'])
                cerebro.adddata(bt.feeds.PandasData(dataname=data))
                results = cerebro.run()
                final_value = cerebro.broker.getvalue()
                st.metric("Backtest Result", f"${final_value:,.2f}", 
                         f"{(final_value/10000-1)*100:.2f}%")

if __name__ == "__main__":
    main()
