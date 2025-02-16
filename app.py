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

    def calculate_position_size(self, entry_price, atr, volatility):
        risk_per_share = entry_price * 0.01
        position_size = (self.portfolio_value * 0.01) / risk_per_share
        volatility_adjustment = 1 / (1 + volatility)
        return int(position_size * volatility_adjustment)

    def backtest(self, df):
        df = df.copy()
        df['Position'] = df['Signal'].diff()
        df['Shares'] = 0
        df['Portfolio_Value'] = self.portfolio_value
        
        for i in df[df['Position'] != 0].index:
            row = df.loc[i]
            if row['Position'] == 1:
                shares = self.calculate_position_size(row['Price'], row['ATR_14'], row['Volatility_21'])
                df.at[i, 'Shares'] = shares
                self.portfolio_value -= shares * row['Price']
            elif row['Position'] == -1:
                prev_shares = df.at[df.index[df.index.get_loc(i)-1], 'Shares']
                self.portfolio_value += prev_shares * row['Price']
                df.at[i, 'Shares'] = 0
            
            df.at[i, 'Portfolio_Value'] = self.portfolio_value + (df.at[i, 'Shares'] * df.at[i, 'Price'])
        
        df['Portfolio_Value'] = df['Portfolio_Value'].ffill().fillna(self.portfolio_value)
        return df

    def calculate_rsi(self, series, period):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean().replace(0, np.nan)
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, df):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Price'].shift())
        low_close = np.abs(df['Low'] - df['Price'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(14, min_periods=1).mean()

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
        
        returns = df['Portfolio_Value'].pct_change().dropna()
        sharpe_ratio = (returns.mean() * TRADING_DAYS_YEAR - RISK_FREE_RATE) / (returns.std() * np.sqrt(TRADING_DAYS_YEAR))
        max_drawdown = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1).min()
        var = rm.calculate_var(returns)
        cvar = rm.calculate_cvar(returns)
        mc_results = rm.monte_carlo_sim(returns)
        
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
    
    except Exception as e:
        st.error(f"System Error: {str(e)}")
        st.stop()

    st.markdown("---")
    st.write("**Disclaimer:** This is an educational tool. Past performance ≠ future results.")

if __name__ == "__main__":
    main()
