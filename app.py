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

def create_initial_data():
    """Create realistic sample data for initial training"""
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
        if self.trade_counter >= 5:
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
            if (self.data.close[0] < self.position.price * (1 - RISK_PARAMS['stop_loss_pct']) or 
                self.data.close[0] > self.position.price * (1 + RISK_PARAMS['take_profit_pct'])):
                self.sell()

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
    """Fixed data fetching with proper session handling"""
    try:
        session = requests.Session()
        session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        
        data = yf.download(
            tickers=pair,
            period='1d',
            interval='5m',
            progress=False,
            session=session
        )
        
        if not data.empty:
            return data[['Open', 'High', 'Low', 'Close', 'Volume']].ffill()
            
    except Exception as e:
        st.error(f"Yahoo Error: {str(e)}")

    try:
        coin_map = {'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'XRP-USD': 'ripple'}
        response = requests.get(
            f'https://api.coingecko.com/api/v3/coins/{coin_map[pair]}/ohlc?vs_currency=usd&days=1',
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=15
        )
        
        if response.status_code == 200:
            ohlc = response.json()
            if isinstance(ohlc, list):
                df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df['volume'] = np.nan
                return df.rename(columns={
                    'open': 'Open', 'high': 'High', 
                    'low': 'Low', 'close': 'Close'
                }).ffill()
                
    except Exception as e:
        st.error(f"CoinGecko Error: {str(e)}")
    
    return pd.DataFrame()

def main():
    st.set_page_config(page_title="Crypto Trading Bot", layout="wide")
    
    st.sidebar.header("Trading Controls")
    selected_pair = st.sidebar.selectbox("Asset Pair", CRYPTO_PAIRS)
    
    if st.sidebar.button("🔄 Update Market Data") and \
        (time.time() - st.session_state.bot_state['last_api_call'] > API_COOLDOWN):
        
        with st.spinner("Fetching real-time data..."):
            try:
                new_data = fetch_market_data(selected_pair)
                if not new_data.empty:
                    st.session_state.bot_state['historical_data'][selected_pair] = new_data
                    st.session_state.bot_state['using_sample_data'] = False
                    st.session_state.bot_state['last_api_call'] = time.time()
                    st.success("Data updated successfully!")
                else:
                    st.warning("No new data received")
            except Exception as e:
                st.error(f"Update failed: {str(e)}")

    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("Account Balance", f"${st.session_state.bot_state['capital']:,.2f}")
        st.write("### Risk Management")
        st.write(f"Max Risk per Trade: {RISK_PARAMS['max_risk_per_trade']*100}%")
        st.write(f"Stop Loss: {RISK_PARAMS['stop_loss_pct']*100}%")
        st.write(f"Take Profit: {RISK_PARAMS['take_profit_pct']*100}%")
    
    with col2:
        data = st.session_state.bot_state['historical_data'].get(selected_pair)
        if data is not None and not data.empty:
            data['RSI'] = RSIIndicator(data['Close']).rsi()
            data['MACD'] = MACD(data['Close']).macd()
            bb = BollingerBands(data['Close'])
            data['BB_Upper'] = bb.bollinger_hband()
            data['BB_Lower'] = bb.bollinger_lband()
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ))
            fig.add_trace(go.Scatter(
                x=data.index, 
                y=data['BB_Upper'],
                line=dict(color='rgba(255, 0, 0, 0.3)'),
                name='Bollinger Upper'
            ))
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_Lower'],
                line=dict(color='rgba(0, 255, 0, 0.3)'),
                name='Bollinger Lower'
            ))
            fig.update_layout(
                height=600,
                title=f"{selected_pair} Market Analysis",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button("Run Backtest"):
                try:
                    cerebro = bt.Cerebro()
                    cerebro.addstrategy(EnhancedStrategy)
                    cerebro.broker.setcash(st.session_state.bot_state['capital'])
                    cerebro.adddata(bt.feeds.PandasData(dataname=data))
                    results = cerebro.run()
                    final_value = cerebro.broker.getvalue()
                    returns = (final_value/st.session_state.bot_state['capital']-1)*100
                    st.metric("Backtest Results", 
                             f"${final_value:,.2f}", 
                             f"{returns:.2f}%")
                except Exception as e:
                    st.error(f"Backtest failed: {str(e)}")

if __name__ == "__main__":
    main()
