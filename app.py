import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import warnings
import os

# Configuration
st.set_page_config(layout="wide", page_title="Pro Trading System")
plt.style.use("ggplot")
pd.set_option('mode.chained_assignment', None)
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(filename="trading_bot.log", level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# Constants
TRADING_DAYS_YEAR = 252
RISK_FREE_RATE = 0.04


class TradingBot:
    def __init__(self, ticker='AAPL', portfolio_value=100000, leverage=5, sma_window=50,
                 stop_loss_pct=0.05, take_profit_pct=0.10, years=5):
        self.ticker = ticker
        self.portfolio_value = portfolio_value
        self.leverage = leverage
        self.sma_window = sma_window
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.years = years

    @st.cache_data(ttl=3600, allow_output_mutation=True)  # Fixed cache issue
    def fetch_data(self, ticker, years):
        """Fetch and align data with a complete business-day index."""
        end = datetime.today()
        start = end - timedelta(days=years * 365)
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, threads=True)
            if df.empty:
                st.error(f"No data found for ticker: {ticker}")
                return None  # Avoid st.stop() inside cache function
            
            df = df[['Close', 'Volume', 'High', 'Low']].rename(columns={'Close': 'Price'})
            full_index = pd.date_range(start=start, end=end, freq='B')
            df = df.reindex(full_index).ffill().dropna()
            return df
        except Exception as e:
            st.error(f"Failed to fetch data: {str(e)}")
            return None  # Prevent Streamlit crashes

    def calculate_features(self, df):
        """Calculate technical indicators."""
        df = df.copy()
        df['SMA_50'] = df['Price'].rolling(50, min_periods=1).mean()
        df['SMA_200'] = df['Price'].rolling(200, min_periods=1).mean()
        df['RSI_14'] = self.calculate_rsi(df['Price'], 14)
        df['ATR_14'] = self.calculate_atr(df)
        df['Volume_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['Daily_Return'] = df['Price'].pct_change()
        df['Volatility_21'] = df['Daily_Return'].rolling(21, min_periods=1).std() * np.sqrt(TRADING_DAYS_YEAR)
        return df.dropna()

    def generate_signals(self, df):
        """Generate trading signals."""
        df = df.copy()
        df['Signal'] = np.where(
            (df['Price'] > df['SMA_50']) &
            (df['RSI_14'] > 30) &
            (df['Volume'] > df['Volume_MA_20']) &
            (df.index.weekday < 5), 1, 0
        )
        df['Signal'] = df['Signal'].shift(1).fillna(0)
        return df

    def backtest(self, df):
        """Run backtest with proper position sizing."""
        df = df.copy()
        df['Position'] = df['Signal'].diff().fillna(0)
        df['Shares'] = 0
        df['Portfolio_Value'] = self.portfolio_value

        for i in df[df['Position'] != 0].index:
            row = df.loc[i]
            if row['Position'] == 1:
                shares = self.calculate_position_size(row['Price'], row['ATR_14'], row['Volatility_21'])
                df.at[i, 'Shares'] = shares
                self.portfolio_value -= shares * row['Price']
            elif row['Position'] == -1:
                prev_shares = df.loc[:i, 'Shares'].iloc[-1] if i > 0 else 0
                self.portfolio_value += prev_shares * row['Price']
                df.at[i, 'Shares'] = 0
            df.at[i, 'Portfolio_Value'] = self.portfolio_value + (df.at[i, 'Shares'] * row['Price'])

        df['Portfolio_Value'] = df['Portfolio_Value'].ffill().fillna(self.portfolio_value)
        return df

    def calculate_rsi(self, series, period=14):
        """Calculate RSI with proper NaN handling."""
        delta = series.diff(1).dropna()
        gain, loss = delta.copy(), delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = abs(loss.rolling(window=period, min_periods=1).mean().replace(0, np.nan))
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, df):
        """Calculate ATR."""
        tr = pd.concat([
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Price'].shift()),
            np.abs(df['Low'] - df['Price'].shift())
        ], axis=1).max(axis=1)
        return tr.rolling(14, min_periods=1).mean()

    def calculate_position_size(self, entry_price, atr, volatility):
        """Calculate position size with volatility adjustment."""
        risk_per_share = entry_price * 0.01
        position_size = (self.portfolio_value * 0.01) / risk_per_share
        volatility_adjustment = 1 / (1 + volatility)
        return int(position_size * volatility_adjustment)

    def run_cycle(self):
        """Run one trading cycle."""
        df = self.fetch_data(self.ticker, self.years)
        if df is None:  # Prevent crashes
            return None, None
        df = self.calculate_features(df)
        df = self.generate_signals(df)
        df = self.backtest(df)
        return df


# ======================
# Streamlit App Interface
# ======================
def main():
    st.title("Professional Trading System")
    bot = TradingBot()

    with st.sidebar:
        st.header("Configuration")
        ticker = st.text_input("Ticker", "AAPL").upper()
        years = st.slider("Backtest Years", 1, 10, 5)
        bot.ticker = ticker
        bot.years = years

    df = bot.run_cycle()
    if df is None:
        st.error("Data could not be fetched. Please check the ticker symbol or try again later.")
        return

    st.subheader("Trading Performance")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Price'], label='Price', color='black')
    ax.plot(df.index, df['SMA_50'], label='50-day SMA', color='blue', linestyle='--')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.metric("Final Portfolio Value", f"${df['Portfolio_Value'].iloc[-1]:,.2f}")
    st.metric("Total Return", f"{(df['Portfolio_Value'].iloc[-1] / 100000 - 1) * 100:.1f}%")

if __name__ == "__main__":
    main()
