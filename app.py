#!/usr/bin/env python3
"""
Autonomous Adaptive Trading System – Streamlit Version (Final Revised with Uniform Data Preparation and Debug Logging)

Features:
  - Automatically installs/upgrades required packages.
  - Fetches free, up-to-date AAPL data from Yahoo Finance.
  - Prepares uniform data by converting the index to datetime, sorting, removing duplicates,
    and reindexing to a complete business-day range (forward-filling missing prices).
  - Computes a 50-day SMA and generates a binary signal (shifted by one day).
  - Calculates daily returns from the uniform data.
  - Reindexes and converts the 'signal' column so that it is a one-dimensional float Series matching the daily returns.
  - Uses pandas’ .mul() method with fill_value=0 to multiply daily returns and signals.
  - Logs and displays debugging information (types, shapes, head of Series) to help diagnose alignment issues.
  - Calculates strategy and cumulative returns.
  - Generates a trade recommendation based on dynamic position sizing.
  - Saves simulation results to a CSV file.
  - Displays an interactive plot and trade recommendation via Streamlit.
  - Includes hidden unit tests (triggered with ?run_tests=true).

DISCLAIMER:
  This system is experimental and uses leverage. No system is foolproof.
  Use with extreme caution and only for educational purposes.
"""

import sys, subprocess, threading, time, os, logging

def install_and_upgrade(packages):
    """Automatically install or upgrade each package in the list."""
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        else:
            print(f"Upgrading {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])

# List of required packages (tkinter is part of the standard library)
required_packages = ["yfinance", "pandas", "numpy", "matplotlib", "schedule", "streamlit"]
install_and_upgrade(required_packages)

# --- Import Libraries ---
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from math import floor
from datetime import datetime, timedelta
import schedule  # For scheduling periodic updates
import unittest

# Set up logging for troubleshooting
logging.basicConfig(filename="trading_bot.log", level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------
# Data Preparation Function
# ---------------------
def prepare_uniform_data(stock_symbol, start_date, end_date, output_file="uniform_data.csv"):
    """
    Fetch raw data for the given stock symbol from Yahoo Finance, then
    convert the index to a complete business-day date range, forward-fill missing prices,
    and save the uniform data to a CSV file.
    """
    df = fetch_stock_data(stock_symbol, start_date, end_date)
    # Ensure the index is datetime, sorted, and duplicates are removed
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    # Create a complete business-day date range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    df = df.reindex(full_range)
    df['price'] = df['price'].fillna(method='ffill')
    df.to_csv(output_file)
    logging.info(f"Uniform data saved to {output_file} from {df.index.min()} to {df.index.max()}")
    return df

# ---------------------
# Core Trading Functions
# ---------------------
def fetch_stock_data(stock_symbol, start_date, end_date):
    """
    Fetch historical daily data for the given stock symbol from Yahoo Finance.
    Use 'Adj Close' if available; otherwise, use 'Close'.
    """
    logging.info(f"Fetching data for {stock_symbol} from {start_date} to {end_date}")
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data fetched. Check the stock symbol and date range.")
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data = data[~data.index.duplicated(keep='first')]
    if 'Adj Close' in data.columns:
        df = data[['Adj Close']].rename(columns={'Adj Close': 'price'})
    elif 'Close' in data.columns:
        df = data[['Close']].rename(columns={'Close': 'price'})
    else:
        raise ValueError("Neither 'Adj Close' nor 'Close' found in the data.")
    return df

def calculate_sma(series, window):
    """Calculate the Simple Moving Average (SMA) of a series."""
    return series.rolling(window=window, min_periods=1).mean()

def generate_signal(df, sma_window=50):
    """
    Generate a binary signal:
      - 1 if price > 50-day SMA,
      - 0 otherwise.
    The signal is shifted by one day to avoid future data.
    """
    df['SMA'] = calculate_sma(df['price'], sma_window)
    df['signal'] = np.where(df['price'] > df['SMA'], 1, 0)
    df['signal'] = df['signal'].shift(1).fillna(0)
    return df

def dynamic_position_size(current_price, stop_loss_pct, portfolio_value, risk_pct=0.01):
    """
    Determine the number of shares to buy so that the risk per trade (current_price * stop_loss_pct)
    is only risk_pct of the portfolio.
    """
    risk_per_share = current_price * stop_loss_pct
    if risk_per_share <= 0:
        return 0
    risk_amount = portfolio_value * risk_pct
    shares = floor(risk_amount / risk_per_share)
    return shares

def calculate_trade_recommendation(df, portfolio_value=10000, leverage=5, stop_loss_pct=0.05, take_profit_pct=0.10):
    """
    Based on the latest data, if the signal is 1, recommend a BUY trade.
    Calculates the number of shares using dynamic position sizing and applies leverage.
    """
    latest = df.iloc[-1]
    current_price = latest['price']
    signal = latest['signal']
    if signal == 1:
        base_shares = dynamic_position_size(current_price, stop_loss_pct, portfolio_value)
        leveraged_shares = base_shares * leverage
        stop_loss = current_price * (1 - stop_loss_pct)
        take_profit = current_price * (1 + take_profit_pct)
        recommendation = {
            'action': 'BUY',
            'stock': 'AAPL',
            'current_price': current_price,
            'num_shares': leveraged_shares,
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    else:
        recommendation = {
            'action': 'HOLD/NO POSITION',
            'stock': 'AAPL',
            'current_price': current_price
        }
    return recommendation

def simulate_leveraged_cumulative_return(df, leverage=5):
    """
    Simulate cumulative return for the leveraged strategy.
    Steps:
      1. Ensure uniform index: convert to datetime, sort, remove duplicates.
      2. Create a complete business-day date range and reindex the DataFrame.
      3. Forward-fill missing prices.
      4. Calculate daily returns.
      5. Reindex the signal to match the complete date range, filling missing values with 0.
      6. Convert daily returns and signal to float Series.
      7. Use pandas' .mul() method with fill_value=0 to multiply.
      8. Calculate strategy and cumulative returns.
      9. Output debugging information.
    """
    # Step 1-3: Uniform index and complete business-day range
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    df = df.reindex(full_range)
    df['price'] = df['price'].fillna(method='ffill')
    
    # Step 4: Calculate daily returns
    df['daily_return'] = df['price'].pct_change().fillna(0).astype(float)
    
    # Step 5: Reindex the 'signal' column; if 'signal' does not exist yet, create it as 0
    if 'signal' not in df.columns:
        df['signal'] = 0.0
    else:
        df['signal'] = df['signal'].astype(float)
    df['signal'] = df['signal'].reindex(full_range, fill_value=0)
    
    # Step 6: Debug output
    st.write("Debug Info - Full date range:", full_range)
    st.write("Debug Info - Daily Return Head:", df['daily_return'].head())
    st.write("Debug Info - Signal Head:", df['signal'].head())
    logging.info(f"Daily return shape: {df['daily_return'].shape}, head: {df['daily_return'].head()}")
    logging.info(f"Signal shape: {df['signal'].shape}, head: {df['signal'].head()}")
    
    # Step 7: Multiply using .mul() with fill_value=0
    multiplied = df['daily_return'].mul(df['signal'], fill_value=0)
    
    # Step 8: Calculate strategy and cumulative returns
    df['strategy_return'] = leverage * multiplied
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    return df

def save_results(df, filename="trading_results.csv"):
    """
    Save the simulation results (timestamp, current price, cumulative return) to a CSV file.
    """
    result = pd.DataFrame({
        "timestamp": [datetime.now()],
        "current_price": [df['price'].iloc[-1]],
        "cumulative_return": [df['cumulative_return'].iloc[-1]]
    })
    if os.path.isfile(filename):
        result.to_csv(filename, mode="a", header=False, index=False)
    else:
        result.to_csv(filename, index=False)
    logging.info("Results saved to " + filename)

def plot_results(df, stock_symbol, start_date, end_date):
    """
    Create a two-panel plot:
      - Top: Stock price and 50-day SMA.
      - Bottom: Cumulative leveraged return.
    Returns the Matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,10), sharex=True)
    ax1.plot(df.index, df['price'], label='Price', color='black')
    ax1.plot(df.index, calculate_sma(df['price'], 50), label='50-day SMA', color='blue', linestyle='--')
    ax1.set_title(f"{stock_symbol} Price and 50-day SMA\n({start_date} to {end_date})")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(df.index, df['cumulative_return'], label='Cumulative Leveraged Return', color='green')
    ax2.set_title("Cumulative Strategy Return")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    return fig

# ---------------------
# Trading Bot Class
# ---------------------
class TradingBot:
    def __init__(self, stock_symbol='AAPL', portfolio_value=10000, leverage=5, sma_window=50,
                 stop_loss_pct=0.05, take_profit_pct=0.10, period_years=3):
        self.stock_symbol = stock_symbol
        self.portfolio_value = portfolio_value
        self.leverage = leverage
        self.sma_window = sma_window
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.period_years = period_years

    def run_cycle(self):
        """
        Run one trading cycle:
          1. Define the date range.
          2. Prepare uniform data.
          3. Generate signals on the uniform data.
          4. Simulate leveraged cumulative return.
          5. Calculate trade recommendation.
          6. Save results.
        """
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365 * self.period_years)).strftime('%Y-%m-%d')
        
        # Prepare uniform data (and save to file for reference)
        uniform_df = prepare_uniform_data(self.stock_symbol, start_date, end_date)
        
        # Generate signal on the uniform data
        uniform_df = generate_signal(uniform_df, sma_window=self.sma_window)
        
        # Simulate leveraged cumulative return on the uniform data
        uniform_df = simulate_leveraged_cumulative_return(uniform_df, leverage=self.leverage)
        
        # Calculate trade recommendation
        recommendation = calculate_trade_recommendation(uniform_df, self.portfolio_value, self.leverage,
                                                        self.stop_loss_pct, self.take_profit_pct)
        # Save results to a CSV file
        save_results(uniform_df)
        return uniform_df, recommendation, start_date, end_date

# ---------------------
# Streamlit App Interface
# ---------------------
st.set_page_config(page_title="Autonomous Adaptive Trading System", layout="wide")
st.title("Autonomous Adaptive Trading System")
st.markdown("""
This system fetches free, up‑to‑date AAPL data, converts it into a uniform dataset with a complete business-day range,
calculates a 50‑day SMA trend signal, simulates an aggressive leveraged strategy with adaptive position sizing,
and displays an interactive plot and trade recommendation.

**DISCLAIMER:** This system is experimental and uses leverage. No system is foolproof.
Use with extreme caution and only for educational purposes.
""")

if st.button("Run Trading Simulation"):
    with st.spinner("Fetching uniform data, running simulation..."):
        try:
            df, rec, start_date, end_date = TradingBot().run_cycle()
            fig = plot_results(df, "AAPL", start_date, end_date)
            st.pyplot(fig)
            if rec['action'] == 'BUY':
                st.success(f"Trade Recommendation for {rec['stock']}:\n"
                           f"Action: BUY\nCurrent Price: ${rec['current_price']:.2f}\n"
                           f"Buy {rec['num_shares']} shares using {rec['leverage']}x leverage\n"
                           f"Stop-Loss: ${rec['stop_loss']:.2f}\n"
                           f"Take-Profit: ${rec['take_profit']:.2f}")
            else:
                st.info(f"Trade Recommendation for {rec['stock']}:\n"
                        f"Action: HOLD/NO POSITION\nCurrent Price: ${rec['current_price']:.2f}")
        except Exception as e:
            st.error(f"Error during simulation: {e}")

if st.button("Show Saved Results"):
    if os.path.isfile("trading_results.csv"):
        saved_df = pd.read_csv("trading_results.csv")
        st.dataframe(saved_df)
    else:
        st.info("No saved results found.")

st.markdown("### About")
st.markdown("""
This application runs autonomously using free historical data.
Each run fetches the most up‑to‑date data (using your device's current date), converts it into a uniform dataset,
executes a full simulation of the trading strategy, saves the results to a CSV file, and displays both visual plots and a clear trade recommendation.
""")

if "run_tests" in st.query_params:
    st.write("Running unit tests...")
    class TestTradingFunctions(unittest.TestCase):
        def setUp(self):
            dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
            prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109], index=dates)
            self.df_test = pd.DataFrame({'price': prices})
    
        def test_calculate_sma(self):
            sma = calculate_sma(self.df_test['price'], window=3)
            expected = (100 + 102 + 101) / 3
            self.assertAlmostEqual(sma.iloc[2], expected)
    
        def test_generate_signal(self):
            df_signal = generate_signal(self.df_test.copy(), sma_window=3)
            self.assertIn('signal', df_signal.columns)
    
        def test_trade_recommendation(self):
            df_signal = generate_signal(self.df_test.copy(), sma_window=3)
            rec = calculate_trade_recommendation(df_signal, portfolio_value=10000, leverage=5)
            self.assertIn('action', rec)
            self.assertIn('stock', rec)
            self.assertIn('current_price', rec)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTradingFunctions)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    st.text("Unit Test Results:")
    st.text(result)
