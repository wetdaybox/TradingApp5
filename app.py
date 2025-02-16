# professional_trading_system.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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
        """Fetch and align data with a complete business day index."""
        end = datetime.today()
        start = end - timedelta(days=years*365)
        
        # Download data and ensure complete business day index
        df = yf.download(ticker, start=start, end=end, progress=False)
        df = df[['Close', 'Volume', 'High', 'Low']].rename(columns={'Close': 'Price'})
        
        # Create a complete business day index
        full_index = pd.date_range(start=start, end=end, freq='B')
        df = df.reindex(full_index).ffill().dropna()
        
        return df

    def calculate_features(self, df):
        """Calculate all technical indicators with proper alignment."""
        df = df.copy()
        
        # Rolling calculations with min_periods=1 to avoid NaNs
        df['SMA_50'] = df['Price'].rolling(50, min_periods=1).mean()
        df['SMA_200'] = df['Price'].rolling(200, min_periods=1).mean()
        df['RSI_14'] = self.calculate_rsi(df['Price'], 14)
        df['ATR_14'] = self.calculate_atr(df)
        df['Volume_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['Daily_Return'] = df['Price'].pct_change()
        df['Volatility_21'] = df['Daily_Return'].rolling(21, min_periods=1).std() * np.sqrt(252)
        
        return df.dropna()

    def generate_signals(self, df):
        """Generate trading signals with explicit alignment."""
        df = df.copy()
        
        # Align all conditions before combining
        price_cond = df['Price'] > df['SMA_50']
        rsi_cond = df['RSI_14'] > 30
        volume_cond = df['Volume'] > df['Volume_MA_20']
        weekday_cond = df.index.weekday < 5
        
        # Combine conditions
        df['Signal'] = np.where(
            price_cond & rsi_cond & volume_cond & weekday_cond,
            1, 0
        )
        
        # Shift signals to avoid look-ahead bias
        df['Signal'] = df['Signal'].shift(1).fillna(0)
        return df

    def backtest(self, df):
        """Run backtest with proper position sizing."""
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
        """Calculate RSI with proper NaN handling."""
        delta = series.diff(1).dropna()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean().replace(0, np.nan)
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, df):
        """Calculate ATR with proper alignment."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Price'].shift())
        low_close = np.abs(df['Low'] - df['Price'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(14, min_periods=1).mean()

    def calculate_position_size(self, entry_price, atr, volatility):
        """Calculate position size with volatility adjustment."""
        risk_per_share = entry_price * 0.01
        position_size = (self.portfolio_value * 0.01) / risk_per_share
        volatility_adjustment = 1 / (1 + volatility)
        return int(position_size * volatility_adjustment)

# ======================
# Streamlit Interface
# ======================
def main():
    st.title("Professional Trading System")
    ts = TradingSystem()
    
    with st.sidebar:
        st.header("Configuration")
        ticker = st.text_input("Ticker", "AAPL").upper()
        years = st.slider("Backtest Years", 1, 10, 5)
        
    try:
        df = ts.fetch_data(ticker, years)
        df = ts.calculate_features(df)
        df = ts.generate_signals(df)
        df = ts.backtest(df)
        
        # Performance Visualization
        st.subheader("Trading Performance")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Portfolio_Value'], label='Portfolio Value')
        ax.plot(df.index, df['Price'], label='Price', alpha=0.5)
        ax.set_title(f"{ticker} Strategy Backtest")
        ax.legend()
        st.pyplot(fig)
        
        # Key Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Portfolio Value", f"${df['Portfolio_Value'].iloc[-1]:,.2f}")
            st.metric("Maximum Drawdown", f"{(df['Portfolio_Value'].min()/df['Portfolio_Value'].max()-1)*100:.1f}%")
        
        with col2:
            st.metric("Total Return", f"{(df['Portfolio_Value'].iloc[-1]/100000-1)*100:.1f}%")
            st.metric("Volatility", f"{df['Volatility_21'].mean()*100:.1f}%")
        
    except Exception as e:
        st.error(f"System Error: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()
