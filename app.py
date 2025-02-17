import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import warnings
import os

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Professional Trading System")
plt.style.use("ggplot")
pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    filename="trading_bot.log",
    level=logging.DEBUG,  # Set to DEBUG to capture detailed information
    format="%(asctime)s [%(levelname)s] %(message)s"
)

TRADING_DAYS_YEAR = 252

# -----------------------------------------------------------------------------
# Data Collection and Uniformization Module
# -----------------------------------------------------------------------------
def collect_uniform_data(ticker, years):
    """
    Download historical data for `ticker` over the past `years` years,
    reindex to a complete business-day DataFrame, forward-fill missing data,
    and save it to a CSV file.
    """
    try:
        end = datetime.today()
        start = end - timedelta(days=years * 365)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        df = df[['Close', 'Volume', 'High', 'Low']].rename(columns={'Close': 'Price'})
        full_index = pd.date_range(start=start, end=end, freq='B')
        df = df.reindex(full_index).ffill().dropna()
        df.to_csv("uniform_data.csv")
        logging.info(f"Data for {ticker} collected and saved to uniform_data.csv.")
        return df
    except Exception as e:
        logging.error(f"Error in collect_uniform_data: {e}")
        raise

@st.cache_data
def cached_uniform_data(ticker, years):
    """Cache the uniform data so repeated calls do not re-download."""
    try:
        if os.path.exists("uniform_data.csv"):
            st.write("Loading uniform data from file...")
            df = pd.read_csv("uniform_data.csv", index_col=0, parse_dates=True)
            # Ensure uniformity
            full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
            if not df.index.equals(full_index):
                st.write("Reindexing uniform data...")
                df = df.reindex(full_index).ffill().dropna()
            logging.info(f"Uniform data for {ticker} loaded from file.")
            return df
        else:
            st.write("Downloading and uniformizing data...")
            return collect_uniform_data(ticker, years)
    except Exception as e:
        logging.error(f"Error in cached_uniform_data: {e}")
        raise

# -----------------------------------------------------------------------------
# Trading Bot Class
# -----------------------------------------------------------------------------
class TradingBot:
    def __init__(self, ticker='AAPL', portfolio_value=100000, leverage=5, sma_window=50,
                 stop_loss_pct=0.05, take_profit_pct=0.10, years=5):
        self.ticker = ticker
        self.initial_portfolio = portfolio_value
        self.portfolio_value = portfolio_value
        self.leverage = leverage
        self.sma_window = sma_window
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.years = years
        self.position = 0

    def calculate_features(self, df):
        """Calculate technical indicators on the uniform data."""
        try:
            df = df.copy()
            full_index = df.index
            df['SMA_50'] = df['Price'].rolling(50, min_periods=1).mean().reindex(full_index, method='ffill')
            df['SMA_200'] = df['Price'].rolling(200, min_periods=1).mean().reindex(full_index, method='ffill')
            df['RSI_14'] = self.calculate_rsi(df['Price'], 14).reindex(full_index, method='ffill')
            df['ATR_14'] = self.calculate_atr(df).reindex(full_index, method='ffill')
            df['Volume_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean().reindex(full_index, method='ffill')
            df['Daily_Return'] = df['Price'].pct_change().reindex(full_index, fill_value=0)
            df['Volatility_21'] = df['Daily_Return'].rolling(21, min_periods=1).std().reindex(full_index, fill_value=0) * np.sqrt(TRADING_DAYS_YEAR)
            logging.info("Features calculated successfully.")
            return df.dropna()
        except Exception as e:
            logging.error(f"Error in calculate_features: {e}")
            raise

    def calculate_rsi(self, series, period=14):
        """Calculate Relative Strength Index (RSI) on a given series."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, df, period=14):
        """Calculate the Average True Range (ATR)."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Price'].shift())
        low_close = np.abs(df['Low'] - df['Price'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def generate_signals(self, df):
        """Generate trading signals ensuring uniform index across all Series."""
        try:
            df = df.copy()
            full_index = df.index
            price_cond = (df['Price'] > df['SMA_50']).reindex(full_index, fill_value=False)
            rsi_cond = (df['RSI_14'] > 30).reindex(full_index, fill_value=False)
            volume_cond = (df['Volume'] > df['Volume_MA_20']).reindex(full_index, fill_value=False)
            weekday_cond = df.index.weekday < 5  # Business days
            df['Signal'] = np.where(price_cond & rsi_cond & volume_cond & weekday_cond, 1, 0)
            df['Signal'] = df['Signal'].shift(1).fillna(0)
            logging.info("Signals generated successfully.")
            return df
        except Exception as e:
            logging.error(f"Error in generate_signals: {e}")
            raise

# -----------------------------------------------------------------------------
# Streamlit App
# -----------------------------------------------------------------------------
st.title("Professional Trading System")

ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL").upper()
years = st.sidebar.slider("Select Data Range (Years)", 1, 10, 5)
portfolio_value = st.sidebar.number_input("Starting Portfolio Value ($)", 10000)

if st.sidebar.button("Run Trading Bot"):
    st.write("Downloading and processing data...")
    try:
        df = cached_uniform_data(ticker, years)
        bot = TradingBot(ticker, portfolio_value)
        df = bot.calculate_features(df)
        df = bot.generate_signals(df)

        st.write("Processed Data Preview:")
        st.dataframe(df.tail())

        # Plot signals
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Price'], label="Price", color='blue')
        ax.scatter(df.index[df['Signal'] == 1], df['Price'][df['Signal'] == 1], label="Buy Signal", marker="^", color='green')
        ax.set_title(f"Trading Signals for {ticker}")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Unhandled error: {e}")
