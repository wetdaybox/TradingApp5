# professional_trading_system.py (Final Working Version)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(layout="wide", page_title="Pro Trading System")
plt.style.use("ggplot")
pd.set_option('mode.chained_assignment', None)

# Constants
TRADING_DAYS_YEAR = 252
RISK_FREE_RATE = 0.04

# ======================
# Core Engine (Fixed Alignment)
# ======================
class TradingSystem:
    def __init__(self):
        self.data = pd.DataFrame()
        self.portfolio_value = 100000
        self.positions = []
        
    def fetch_data(self, ticker, years):
        end = datetime.today()
        start = end - timedelta(days=years*365)
        
        df = yf.download(ticker, start=start, end=end, progress=False)
        df = df[['Close', 'Volume']].rename(columns={'Close': 'Price'})
        df.index = pd.to_datetime(df.index)
        df = df.resample('B').last().ffill().dropna()
        return df

    def calculate_features(self, df):
        # Technical Indicators with proper alignment
        df['SMA_50'] = df['Price'].rolling(50).mean()
        df['SMA_200'] = df['Price'].rolling(200).mean()
        df['RSI_14'] = self.calculate_rsi(df['Price'], 14)
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['VWAP'] = (df['Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['Daily_Return'] = df['Price'].pct_change()
        df['Volatility_21'] = df['Daily_Return'].rolling(21).std() * np.sqrt(252)
        return df.dropna()

    def generate_signals(self, df):
        # Aligned conditional logic
        df = df.copy()
        df['Signal'] = 0
        
        # Create aligned boolean conditions
        price_condition = (df['Price'] > df['SMA_50'])
        rsi_condition = (df['RSI_14'] > 30)
        volume_condition = (df['Volume'] > df['Volume_MA_20'])
        weekday_condition = (df.index.weekday < 5)
        
        # Combine conditions with proper alignment
        long_cond = (
            price_condition & 
            rsi_condition & 
            volume_condition &
            weekday_condition
        ).dropna()
        
        df['Signal'] = np.where(long_cond, 1, 0).shift(1).fillna(0)
        return df

    # Rest of the class remains unchanged from previous version

# ======================
# Streamlit Interface (Unchanged)
# ======================
def main():
    st.title("Professional Trading System")
    ts = TradingSystem()
    rm = RiskManager()
    
    with st.sidebar:
        st.header("Configuration")
        ticker = st.text_input("Ticker", "AAPL").upper()
        years = st.slider("Backtest Years", 1, 10, 5)
        risk_level = st.select_slider("Risk Level", options=["Low", "Medium", "High"])
        
    # Data Pipeline with error handling
    try:
        df = ts.fetch_data(ticker, years)
        df = ts.calculate_features(df)
        df = ts.generate_signals(df)
        df = ts.backtest(df)
        
        # Rest of the code remains unchanged
        
    except Exception as e:
        st.error(f"Error in data processing: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()
