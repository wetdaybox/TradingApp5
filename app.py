import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import backtrader as bt
import requests
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
API_COOLDOWN = 65

def create_initial_data():
    """Create realistic sample data with proper array dimensions"""
    base_prices = {'BTC-USD': 45000, 'ETH-USD': 2500, 'XRP-USD': 0.55}
    sample_data = {}
    
    for pair in CRYPTO_PAIRS:
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=300, freq='5min')  # Increased to 300 periods
        base_price = base_prices[pair]
        volatility = 0.015 + (0.005 if pair == 'XRP-USD' else 0)
        
        price_changes = np.cumsum(volatility * np.random.randn(300)) / 100
        close = base_price * (1 + price_changes)
        open_price = close * 0.998
        high = close * 1.005
        low = close * 0.995
        
        sample_data[pair] = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': np.random.randint(1000, 5000, 300)
        }, index=dates).iloc[:288]  # Ensure exact 288 periods
    
    return sample_data

class RobustStrategy(bt.Strategy):
    params = (
        ('sma_fast', 20),
        ('sma_slow', 50),
        ('rsi_period', 14),
        ('atr_period', 14),
        ('risk_per_trade', 0.01)
    )

    def __init__(self):
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.p.sma_fast)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.p.sma_slow)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if (self.sma_fast > self.sma_slow and 
                self.rsi < 35 and 
                len(self) > self.p.atr_period):  # Ensure enough bars
                
                risk_amount = self.broker.getvalue() * self.p.risk_per_trade
                size = risk_amount / (self.atr[0] * 2)
                self.buy(size=size)
        else:
            if (self.data.close[0] < self.position.price * (1 - RISK_PARAMS['stop_loss_pct']) or 
                self.data.close[0] > self.position.price * (1 + RISK_PARAMS['take_profit_pct'])):
                self.sell()

# Session state initialization
if 'bot_state' not in st.session_state:
    st.session_state.update({
        'bot_state': {
            'positions': {},
            'capital': 10000,
            'historical_data': create_initial_data(),
            'last_api_call': 0,
            'update_in_progress': False,
            'using_sample_data': True
        }
    })

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
def fetch_market_data(pair):
    """Robust data fetching with array bounds checking"""
    try:
        session = requests.Session()
        session.headers['User-Agent'] = 'Mozilla/5.0'
        
        data = yf.download(
            tickers=pair,
            period='2d',  # Get extra data for buffer
            interval='5m',
            progress=False,
            session=session
        )
        
        if not data.empty:
            # Ensure proper data length and columns
            return data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-288:].reset_index(drop=True)
            
    except Exception as e:
        st.error(f"Yahoo Error: {str(e)}")

    try:
        coin_map = {'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'XRP-USD': 'ripple'}
        response = requests.get(
            f'https://api.coingecko.com/api/v3/coins/{coin_map[pair]}/ohlc?vs_currency=usd&days=2',
            timeout=15
        )
        
        if response.status_code == 200:
            ohlc = response.json()
            if isinstance(ohlc, list) and len(ohlc) > 0:
                df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df.rename(columns={
                    'open': 'Open', 'high': 'High',
                    'low': 'Low', 'close': 'Close'
                }).assign(Volume=np.nan).iloc[-288:]
                
    except Exception as e:
        st.error(f"CoinGecko Error: {str(e)}")
    
    return pd.DataFrame()

def main():
    st.set_page_config(page_title="Crypto Trading Bot", layout="wide")
    
    st.sidebar.header("Controls")
    selected_pair = st.sidebar.selectbox("Asset", CRYPTO_PAIRS)
    
    if st.sidebar.button("🔄 Update Data"):
        with st.spinner("Updating..."):
            new_data = fetch_market_data(selected_pair)
            if not new_data.empty and len(new_data) >= 100:
                st.session_state.bot_state['historical_data'][selected_pair] = new_data
                st.session_state.bot_state['using_sample_data'] = False
                st.success("Data updated!")
            else:
                st.warning("Using existing data - update failed")

    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("Balance", f"${st.session_state.bot_state['capital']:,.2f}")
        st.write("### Risk Parameters")
        st.write(f"Max Risk/Trade: {RISK_PARAMS['max_risk_per_trade']*100}%")
        st.write(f"Stop Loss: {RISK_PARAMS['stop_loss_pct']*100}%")
        st.write(f"Take Profit: {RISK_PARAMS['take_profit_pct']*100}%")
    
    with col2:
        data = st.session_state.bot_state['historical_data'].get(selected_pair)
        if data is not None and len(data) >= 100:
            # Ensure indicator calculations
            data = data.copy()
            data['RSI'] = RSIIndicator(data['Close']).rsi()
            data['MACD'] = MACD(data['Close']).macd()
            bb = BollingerBands(data['Close'])
            data['BB_Upper'] = bb.bollinger_hband()
            data['BB_Lower'] = bb.bollinger_lband()
            
            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close']
            ))
            fig.update_layout(height=600, title=f"{selected_pair} Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Backtest execution
            if st.button("Run Backtest"):
                try:
                    cerebro = bt.Cerebro()
                    cerebro.addstrategy(RobustStrategy)
                    
                    # Proper data feed initialization
                    datafeed = bt.feeds.PandasData(
                        dataname=data,
                        datetime=None,
                        open=0,
                        high=1,
                        low=2,
                        close=3,
                        volume=4,
                        openinterest=-1
                    )
                    
                    cerebro.adddata(datafeed)
                    cerebro.broker.setcash(st.session_state.bot_state['capital'])
                    cerebro.broker.setcommission(commission=0.001)
                    
                    results = cerebro.run()
                    final_value = cerebro.broker.getvalue()
                    st.success(f"Final Portfolio Value: ${final_value:,.2f}")
                except Exception as e:
                    st.error(f"Backtest failed: {str(e)}")
        else:
            st.warning("Insufficient data for analysis")

if __name__ == "__main__":
    main()
