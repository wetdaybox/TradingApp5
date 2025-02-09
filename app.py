#!/usr/bin/env python3
"""
Autonomous Adaptive Trading System – Streamlit Version (Final with Explicit Alignment)

Features:
  - Automatically installs/upgrades required packages.
  - Fetches free, up‑to‑date AAPL data from Yahoo Finance.
  - Prepares uniform data by converting the index to datetime, sorting it, removing duplicates,
    and reindexing to a full business-day range (forward-filling missing prices).
  - Computes a 50‑day SMA and generates a binary signal (shifted by one day).
  - Calculates daily returns on the uniform data.
  - Reindexes and converts the signal to a one‑dimensional float Series.
  - Uses .align() to explicitly align daily returns and signal, then multiplies them.
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

# List of required packages
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
    Fetch raw data for the given stock symbol from Yahoo Finance, convert the index to datetime,
    sort it, remove duplicates, reindex to a complete business-day range, forward-fill missing prices,
    and save the uniform data to a CSV file.
    """
    df = fetch_stock_data(stock_symbol, start_date, end_date)
    # Convert index to datetime, sort it, and remove duplicates
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    # Create a complete business-day date range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    df = df.reindex(full_range)
    # Forward-fill missing prices
    df['price'] = df['price'].fillna(method='ffill')
    df.to_csv(output_file)
    logging.info(f"Uniform data saved to {output_file} with index from {df.index.min()} to {df.index.max()}")
    return df

# ---------------------
# Core Trading Functions
# ---------------------
def fetch_stock_data(stock_symbol, start_date, end_date):
    """
    Fetch historical daily data for the given stock symbol from Yahoo Finance.
    Uses 'Adj Close' if available; otherwise, uses 'Close'.
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
    Shift the signal by one day to avoid using future data.
    """
    df['SMA'] = calculate_sma(df['price'], sma_window)
    df['signal'] = np.where(df['price'] > df['SMA'], 1, 0)
    df['signal'] = df['signal'].shift(1).fillna(0)
    return df

def dynamic_position_size(current_price, stop_loss_pct, portfolio_value, risk_pct=0.01):
    """Calculate the number of shares such that risk per trade is limited to risk_pct of the portfolio."""
    risk_per_share = current_price * stop_loss_pct
    if risk_per_share <= 0:
        return 0
    risk_amount = portfolio_value * risk_pct
    shares = floor(risk_amount / risk_per_share)
    return shares

def calculate_trade_recommendation(df, portfolio_value=10000, leverage=5, stop_loss_pct=0.05, take_profit_pct=0.10):
    """
    Based on the latest data, if the signal is 1, recommend a BUY trade.
    Use dynamic position sizing and a leverage factor to compute share amounts.
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
    Simulate cumulative return for the leveraged strategy using uniform data.
    Steps:
      1. Ensure df has a complete business-day index (should be done by prepare_uniform_data).
      2. Calculate daily returns.
      3. Reindex the 'signal' column to match the complete date range (fill missing values with 0).
      4. Convert daily returns and signal to one-dimensional float Series.
      5. Align the two Series explicitly using .align() on axis=0.
      6. Multiply them elementwise using the * operator.
      7. Calculate cumulative return.
      8. Output debugging information.
    """
    # Step 2: Calculate daily returns
    df['daily_return'] = df['price'].pct_change().fillna(0).astype(float)
    
    # Step 3: Reindex the 'signal' column to match the DataFrame index (full business days)
    if 'signal' not in df.columns:
        df['signal'] = 0.0
    else:
        df['signal'] = df['signal'].astype(float)
    df['signal'] = df['signal'].reindex(df.index, fill_value=0)
    
    # Step 4 & 5: Convert to Series and align explicitly using .align() on axis=0
    dr = df['daily_return']
    sig = df['signal']
    dr_aligned, sig_aligned = dr.align(sig, axis=0, fill_value=0)
    
    # Debug: Output types, shapes, and heads
    st.write("Debug Info - daily_return type:", type(dr_aligned), "shape:", dr_aligned.shape)
    st.write("Debug Info - signal type:", type(sig_aligned), "shape:", sig_aligned.shape)
    st.write("Debug Info - daily_return head:", dr_aligned.head())
    st.write("Debug Info - signal head:", sig_aligned.head())
    logging.info(f"Aligned daily_return shape: {dr_aligned.shape}, head: {dr_aligned.head()}")
    logging.info(f"Aligned signal shape: {sig_aligned.shape}, head: {sig_aligned.head()}")
    
    # Step 6: Multiply elementwise
    multiplied = dr_aligned * sig_aligned
    # Step 7: Calculate strategy and cumulative returns
    df['strategy_return'] = leverage * multiplied
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    return df

def save_results(df, filename="trading_results.csv"):
    """
    Save simulation results (timestamp, current price, cumulative return) to a CSV file.
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
    Returns a Matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,10), sharex=True)
    ax1.plot(df.index, df['price'], label='Price', color='black')
    # Recalculate SMA on the fly for plotting
    sma = calculate_sma(df['price'], 50)
    ax1.plot(df.index, sma, label='50-day SMA', color='blue', linestyle='--')
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
          3. Generate signal on the uniform data.
          4. Simulate leveraged cumulative return.
          5. Calculate trade recommendation.
          6. Save results.
        """
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365 * self.period_years)).strftime('%Y-%m-%d')
        
        # Step 2: Prepare uniform data and save it to a file
        uniform_df = prepare_uniform_data(self.stock_symbol, start_date, end_date)
        
        # Step 3: Generate signal on the uniform data
        uniform_df = generate_signal(uniform_df, sma_window=self.sma_window)
        
        # Step 4: Simulate leveraged cumulative return on the uniform data
        uniform_df = simulate_leveraged_cumulative_return(uniform_df, leverage=self.leverage)
        
        # Step 5: Calculate trade recommendation
        recommendation = calculate_trade_recommendation(uniform_df, self.portfolio_value, self.leverage,
                                                        self.stop_loss_pct, self.take_profit_pct)
        # Step 6: Save results
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
    with st.spinner("Preparing uniform data and running simulation..."):
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
Each run fetches the most up‑to‑date data (using your device's current date) and converts it into a uniform dataset with a complete business-day range.
It then executes a full simulation of the trading strategy, saves the results to a CSV file for later review,
and displays both visual plots and a clear trade recommendation.
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
