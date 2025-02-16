# professional_trading_system.py
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
# Core Trading Engine
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
        df = df[['Close', 'Volume', 'High', 'Low']].rename(columns={'Close': 'Price'})
        df.index = pd.to_datetime(df.index)
        return df.resample('B').last().ffill().dropna()

    def calculate_features(self, df):
        # Step-by-step feature engineering
        df = df.copy()
        df['SMA_50'] = df['Price'].rolling(50, min_periods=1).mean()
        df['SMA_200'] = df['Price'].rolling(200, min_periods=1).mean()
        df['RSI_14'] = self.calculate_rsi(df['Price'], 14)
        df['ATR_14'] = self.calculate_atr(df)
        df['Volume_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['VWAP'] = (df['Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['Daily_Return'] = df['Price'].pct_change()
        df['Volatility_21'] = df['Daily_Return'].rolling(21).std() * np.sqrt(252)
        return df.dropna()

    def generate_signals(self, df):
        # Explicit conditional logic
        df = df.copy()
        df['price_condition'] = df['Price'] > df['SMA_50']
        df['rsi_condition'] = df['RSI_14'] > 30
        df['volume_condition'] = df['Volume'] > df['Volume_MA_20']
        df['weekday_condition'] = df.index.weekday < 5
        
        df['Signal'] = np.where(
            df['price_condition'] & 
            df['rsi_condition'] & 
            df['volume_condition'] & 
            df['weekday_condition'],
            1, 0
        )
        df['Signal'] = df['Signal'].shift(1).fillna(0)
        return df.drop(columns=['price_condition', 'rsi_condition', 'volume_condition', 'weekday_condition'])

    # Rest of the class remains unchanged (backtest, calculate_rsi, etc.)

# ======================
# Risk Management Module
# ======================
class RiskManager:
    @staticmethod
    def calculate_var(returns, confidence=0.95):
        return np.percentile(returns, 100*(1-confidence))

    @staticmethod
    def calculate_cvar(returns, confidence=0.95):
        var = RiskManager.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    @staticmethod
    def monte_carlo_sim(returns, days=252, simulations=1000):
        log_returns = np.log(1 + returns.dropna())
        mu = log_returns.mean()
        sigma = log_returns.std()
        daily_returns = np.random.normal(mu, sigma, (days, simulations))
        cumulative_returns = np.exp(daily_returns).cumprod(axis=0)
        return np.percentile(cumulative_returns[-1], [5, 50, 95])

# ======================
# Streamlit Interface
# ======================
def main():
    st.title("Professional Trading System")
    ts = TradingSystem()
    rm = RiskManager()
    
    with st.sidebar:
        st.header("Configuration")
        ticker = st.text_input("Ticker", "AAPL").upper()
        years = st.slider("Backtest Years", 1, 10, 5)
        
    try:
        df = ts.fetch_data(ticker, years)
        df = ts.calculate_features(df)
        df = ts.generate_signals(df)
        df = ts.backtest(df)
        
        # Performance metrics and display code remains unchanged
        
    except Exception as e:
        st.error(f"System Error: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()
