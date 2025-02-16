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
RISK_FREE_RATE = 0.04  # 4% annual

# ======================
# Core Engine (Fixed)
# ======================
class TradingSystem:
    def __init__(self):
        self.data = pd.DataFrame()
        self.portfolio_value = 100000  # Starting capital
        self.positions = []
        
    def fetch_data(self, ticker, years):
        end = datetime.today()
        start = end - timedelta(days=years*365)
        
        # Updated to use modern yfinance format
        df = yf.download(ticker, start=start, end=end, progress=False)
        df = df[['Close', 'Volume']].rename(columns={'Close': 'Price'})  # Fixed column name
        df.index = pd.to_datetime(df.index)
        
        # Clean and regularize data
        df = df.resample('B').last()
        df = df.ffill().dropna()
        return df

    def calculate_features(self, df):
        # Technical Indicators
        df['SMA_50'] = df['Price'].rolling(50).mean()
        df['SMA_200'] = df['Price'].rolling(200).mean()
        df['RSI_14'] = self.calculate_rsi(df['Price'], 14)
        df['ATR_14'] = self.calculate_atr(df, 14)
        
        # Volume Features
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['VWAP'] = (df['Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Volatility
        df['Daily_Return'] = df['Price'].pct_change()
        df['Volatility_21'] = df['Daily_Return'].rolling(21).std() * np.sqrt(252)
        return df.dropna()

    def generate_signals(self, df):
        # Enhanced Strategy Logic
        df['Signal'] = 0
        long_cond = (
            (df['Price'] > df['SMA_50']) & 
            (df['RSI_14'] > 30) & 
            (df['Volume'] > df['Volume_MA_20']) &
            (df.index.weekday < 5)  # Only trade weekdays
        )
        df['Signal'] = np.where(long_cond, 1, 0).shift(1).fillna(0)
        return df

    def calculate_position_size(self, entry_price, atr, volatility):
        # Professional Position Sizing
        risk_per_share = entry_price * 0.01  # 1% risk per position
        position_size = (self.portfolio_value * 0.01) / risk_per_share
        volatility_adjustment = 1 / (1 + volatility)
        return int(position_size * volatility_adjustment)

    def backtest(self, df):
        # Realistic Backtest Engine
        df['Position'] = df['Signal'].diff()
        df['Shares'] = 0
        df['Portfolio_Value'] = self.portfolio_value
        
        for i in range(1, len(df)):
            if df['Position'].iloc[i] == 1:
                entry_price = df['Price'].iloc[i]
                atr = df['ATR_14'].iloc[i]
                volatility = df['Volatility_21'].iloc[i]
                shares = self.calculate_position_size(entry_price, atr, volatility)
                df['Shares'].iloc[i] = shares
                investment = shares * entry_price
                self.portfolio_value -= investment
                
            elif df['Position'].iloc[i] == -1:
                exit_price = df['Price'].iloc[i]
                shares = df['Shares'].iloc[i-1]
                proceeds = shares * exit_price
                self.portfolio_value += proceeds
                df['Shares'].iloc[i] = 0
                
            df['Portfolio_Value'].iloc[i] = self.portfolio_value + (
                df['Shares'].iloc[i] * df['Price'].iloc[i]
            )
        return df

    def calculate_rsi(self, series, period):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, df, period):
        # Requires High/Low data
        df_ext = yf.download('AAPL', period='1y')  # Get full OHLC data
        high_low = df_ext['High'] - df_ext['Low']
        high_close = np.abs(df_ext['High'] - df_ext['Close'].shift())
        low_close = np.abs(df_ext['Low'] - df_ext['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

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
        log_returns = np.log(1 + returns)
        mu = log_returns.mean()
        sigma = log_returns.std()
        
        results = []
        for _ in range(simulations):
            daily_returns = np.random.normal(mu, sigma, days)
            results.append(np.exp(daily_returns).cumprod()[-1])
        return np.percentile(results, [5, 50, 95])

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
        risk_level = st.select_slider("Risk Level", options=["Low", "Medium", "High"])
        
    # Data Pipeline
    df = ts.fetch_data(ticker, years)
    df = ts.calculate_features(df)
    df = ts.generate_signals(df)
    df = ts.backtest(df)
    
    # Performance Metrics
    returns = df['Portfolio_Value'].pct_change().dropna()
    sharpe_ratio = (returns.mean() * TRADING_DAYS_YEAR - RISK_FREE_RATE) / (returns.std() * np.sqrt(TRADING_DAYS_YEAR))
    max_drawdown = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1).min()
    var = rm.calculate_var(returns)
    cvar = rm.calculate_cvar(returns)
    mc_results = rm.monte_carlo_sim(returns)
    
    # Main Display
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Risk Metrics")
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        st.metric("Max Drawdown", f"{(max_drawdown*100):.1f}%")
        st.metric("VaR (95%)", f"{(var*100):.1f}%")
        st.metric("CVaR (95%)", f"{(cvar*100):.1f}%")
        
        st.subheader("Monte Carlo Simulation")
        st.write(f"5% Worst Case: {(mc_results[0]-1)*100:.1f}%")
        st.write(f"Median Case: {(mc_results[1]-1)*100:.1f}%")
        st.write(f"95% Best Case: {(mc_results[2]-1)*100:.1f}%")
    
    with col2:
        st.subheader("Performance Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Portfolio_Value'], label='Portfolio Value')
        ax.plot(df.index, df['Price'], label='Price', alpha=0.5)
        ax.set_title(f"{ticker} Trading Performance")
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("Position Sizing")
        st.line_chart(df['Shares'])
    
    # Risk Management Report
    with st.expander("Detailed Risk Analysis"):
        st.write("""
        ### Risk Management Framework
        - **Value at Risk (VaR):** Maximum expected loss over 1 day at 95% confidence
        - **Conditional VaR:** Expected loss given we're in the worst 5% of cases
        - **Monte Carlo Simulation:** 1-year forward-looking return distribution
        """)
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Return Distribution")
            st.line_chart(returns.rolling(21).std() * np.sqrt(252))
            
        with col4:
            st.subheader("Volatility Clustering")
            st.line_chart(df['Volatility_21'])
    
    st.markdown("---")
    st.write("**Disclaimer:** This is an educational tool. Past performance ≠ future results.")

if __name__ == "__main__":
    main()
