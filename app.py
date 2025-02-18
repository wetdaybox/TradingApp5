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
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set up logging (DEBUG level captures detailed info)
logging.basicConfig(
    filename="trading_bot.log",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

TRADING_DAYS_YEAR = 252
RISK_FREE_RATE = 0.04

# -----------------------------------------------------------------------------
# Data Collection and Uniformization Module
# -----------------------------------------------------------------------------
def collect_uniform_data(ticker, years):
    """
    Download historical data for the given ticker over the past `years` years,
    reindex it to a complete business-day DataFrame, forward-fill missing data,
    and save it to a CSV file.
    """
    try:
        end = datetime.today()
        start = end - timedelta(days=years * 365)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        # Keep essential columns and rename 'Close' to 'Price'
        df = df[['Close', 'Volume', 'High', 'Low']].rename(columns={'Close': 'Price'})
        full_index = pd.date_range(start=start, end=end, freq='B')
        df = df.reindex(full_index).ffill().dropna()
        df.to_csv("uniform_data.csv")
        logging.info(f"Data for {ticker} collected and saved to uniform_data.csv.")
        return df
    except Exception as e:
        logging.error(f"Error in collect_uniform_data: {e}")
        st.error(f"Data collection error: {e}")
        raise

@st.cache_data
def cached_uniform_data(ticker, years):
    """
    Load uniform data from file if available; otherwise, download and process.
    Ensures the DataFrame has a complete business-day index.
    """
    try:
        if os.path.exists("uniform_data.csv"):
            st.write("Loading uniform data from file...")
            df = pd.read_csv("uniform_data.csv", index_col=0, parse_dates=True)
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
        st.error(f"Uniform data error: {e}")
        raise

# -----------------------------------------------------------------------------
# Trading Bot Class
# -----------------------------------------------------------------------------
class TradingBot:
    def __init__(self, ticker='AAPL', portfolio_value=100000, leverage=5, sma_window=50,
                 stop_loss_pct=0.05, take_profit_pct=0.10, years=5):
        self.ticker = ticker
        self.initial_portfolio = portfolio_value
        self.portfolio_value = portfolio_value  # Tracks cash available
        self.leverage = leverage
        self.sma_window = sma_window
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.years = years

    def fetch_data(self, ticker, years):
        """Wrapper to get cached uniform data."""
        return cached_uniform_data(ticker, years)

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
            st.error(f"Feature calculation error: {e}")
            raise

    def calculate_rsi(self, series, period=14):
        delta = series.diff().dropna()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean().replace(0, np.inf)
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, df, period=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Price'].shift())
        low_close = np.abs(df['Low'] - df['Price'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    def generate_signals(self, df):
        """Generate trading signals."""
        try:
            df = df.copy()
            full_index = df.index
            # Create a Series from the weekday values so all operands have an index
            weekday_series = pd.Series(df.index.weekday, index=full_index)
            df['Signal'] = np.where(
                (df['Price'] > df['SMA_50']) &
                (df['RSI_14'] > 30) &
                (df['Volume'] > df['Volume_MA_20']) &
                (weekday_series < 5),
                1, 0
            )
            df['Signal'] = df['Signal'].shift(1).fillna(0)
            logging.info("Signals generated successfully.")
            return df
        except Exception as e:
            logging.error(f"Error in generate_signals: {e}")
            st.error(f"Signal generation error: {e}")
            raise

    def backtest(self, df):
        """Run backtest with dynamic position sizing and leverage handling."""
        try:
            df = df.copy()
            df['Position'] = df['Signal'].diff().fillna(0)
            df['Shares'] = 0.0
            cash = self.initial_portfolio
            shares_held = 0.0
            portfolio_values = []

            for i, row in df.iterrows():
                current_price = row['Price']
                # Update portfolio value before any trades
                current_value = cash + shares_held * current_price
                portfolio_values.append(current_value)

                if row['Position'] == 1 and shares_held == 0:
                    risk_per_share = current_price * self.stop_loss_pct
                    position_size = (cash * self.leverage * 0.01) / risk_per_share
                    max_shares = (cash * self.leverage) // current_price
                    shares = min(position_size, max_shares)
                    cost = shares * current_price
                    margin_required = cost / self.leverage
                    if margin_required > cash:
                        shares = (cash * self.leverage) // current_price
                        margin_required = (shares * current_price) / self.leverage
                    cash -= margin_required
                    shares_held += shares
                    df.at[i, 'Shares'] = shares
                elif row['Position'] == -1 and shares_held > 0:
                    proceeds = shares_held * current_price
                    cash += proceeds
                    shares_held = 0.0
                    df.at[i, 'Shares'] = 0.0

                df.at[i, 'Portfolio_Value'] = cash + shares_held * current_price

            df['Portfolio_Value'] = portfolio_values
            logging.info("Backtest completed successfully.")
            return df
        except Exception as e:
            logging.error(f"Error in backtest: {e}")
            st.error(f"Backtest error: {e}")
            raise

    def calculate_position_size(self, entry_price, atr, volatility):
        risk_per_share = entry_price * 0.01
        position_size = (self.portfolio_value * 0.01) / risk_per_share
        volatility_adjustment = 1 / (1 + volatility)
        return int(position_size * volatility_adjustment)

    def calculate_trade_recommendation(self, df):
        latest = df.iloc[-1]
        current_price = latest['Price']
        if latest['Signal'] == 1:
            risk_per_share = current_price * 0.01
            position_size = (self.initial_portfolio * 0.01) / risk_per_share
            position_size *= self.leverage
            volatility_adj = 1 / (1 + latest['Volatility_21'])
            shares = int(position_size * volatility_adj)
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
            return {
                'action': 'BUY',
                'stock': self.ticker,
                'current_price': current_price,
                'num_shares': shares,
                'leverage': self.leverage,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        else:
            return {
                'action': 'HOLD/NO POSITION',
                'stock': self.ticker,
                'current_price': current_price
            }

    def save_results(self, df, filename="trading_results.csv"):
        try:
            result = pd.DataFrame({
                "timestamp": [datetime.now()],
                "current_price": [df['Price'].iloc[-1]],
                "portfolio_value": [df['Portfolio_Value'].iloc[-1]]
            })
            if os.path.isfile(filename):
                result.to_csv(filename, mode="a", header=False, index=False)
            else:
                result.to_csv(filename, index=False)
            logging.info("Results saved to " + filename)
        except Exception as e:
            logging.error(f"Error in save_results: {e}")
            raise

    def run_cycle(self):
        """
        Run one trading cycle:
          1. Fetch uniform data.
          2. Calculate technical features.
          3. Generate signals.
          4. Run backtest.
          5. Simulate cumulative returns.
          6. Calculate trade recommendation.
          7. Save results.
        """
        try:
            df_raw = self.fetch_data(self.ticker, self.years)
            if df_raw is None:
                st.error("Data fetch failed.")
                return None, None
            df_feat = self.calculate_features(df_raw)
            df_signals = self.generate_signals(df_feat)
            df_backtest = self.backtest(df_signals)
            df_final = simulate_leveraged_cumulative_return(df_backtest, leverage=self.leverage)
            rec = self.calculate_trade_recommendation(df_final)
            self.save_results(df_final)
            logging.info("Run cycle completed successfully.")
            return df_final, rec
        except Exception as e:
            logging.error(f"Error in run_cycle: {e}")
            st.error(f"Run cycle error: {e}")
            return None, None

# -----------------------------------------------------------------------------
# Simulation of Strategy Returns Module
# -----------------------------------------------------------------------------
def simulate_leveraged_cumulative_return(df, leverage=5):
    """
    Calculate daily returns and strategy returns by forcing uniform multiplication.
    We reset the index on the daily_return and Signal Series to a default integer index,
    multiply them elementwise, and then reconstruct the resulting Series with the original DatetimeIndex.
    This ensures no alignment errors occur.
    """
    try:
        df['daily_return'] = df['Price'].pct_change().fillna(0).astype(float).squeeze()
        if 'Signal' not in df.columns:
            df['Signal'] = 0.0
        else:
            df['Signal'] = df['Signal'].astype(float).squeeze()
        df['Signal'] = df['Signal'].reindex(df.index, fill_value=0)
        
        # Reset index to force default integer alignment
        dr_reset = df['daily_return'].reset_index(drop=True)
        sig_reset = df['Signal'].reset_index(drop=True)
        product = dr_reset * sig_reset
        # Reconstruct the Series with the original DatetimeIndex
        df['strategy_return'] = leverage * pd.Series(product.values, index=df.index)
        df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
        
        st.write("Debug - daily_return shape:", df['daily_return'].shape)
        st.write("Debug - Signal shape:", df['Signal'].shape)
        st.write("Debug - First 5 daily_return values:", df['daily_return'].head())
        st.write("Debug - First 5 Signal values:", df['Signal'].head())
        logging.info(f"daily_return shape: {df['daily_return'].shape}, head: {df['daily_return'].head()}")
        logging.info(f"Signal shape: {df['Signal'].shape}, head: {df['Signal'].head()}")
        
        return df
    except Exception as e:
        logging.error(f"Error in simulate_leveraged_cumulative_return: {e}")
        st.error(f"Strategy return calculation error: {e}")
        raise

# -----------------------------------------------------------------------------
# Plotting Function
# -----------------------------------------------------------------------------
def plot_results(df, ticker, start_date, end_date):
    """Plot Price and cumulative return."""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,6), sharex=True)
        ax1.plot(df.index, df['Price'], label='Price', color='black')
        sma = df['Price'].rolling(50, min_periods=1).mean()
        ax1.plot(df.index, sma, label='50-day SMA', color='blue', linestyle='--')
        ax1.set_title(f"{ticker} Price and 50-day SMA\n({start_date} to {end_date})")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(df.index, df['cumulative_return'], label='Cumulative Return', color='green')
        ax2.set_title("Cumulative Strategy Return")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_results: {e}")
        st.error(f"Plotting error: {e}")
        raise

# -----------------------------------------------------------------------------
# Optional Cleanup Function
# -----------------------------------------------------------------------------
def cleanup_temp_files():
    """Delete temporary files created by the program."""
    for file in ["uniform_data.csv", "trading_results.csv"]:
        if os.path.isfile(file):
            st.write(f"Deleting temporary file: {file}")
            os.remove(file)
            logging.info(f"Deleted temporary file: {file}")

# -----------------------------------------------------------------------------
# Streamlit App Interface
# -----------------------------------------------------------------------------
def main():
    st.title("Professional Trading System")
    bot = TradingBot()

    with st.sidebar:
        st.header("Configuration")
        ticker = st.text_input("Ticker", "AAPL").upper()
        years = st.slider("Backtest Years", 1, 10, 5)
        leverage = st.slider("Leverage", 1, 10, 5)
        stop_loss = st.slider("Stop Loss (%)", 1.0, 10.0, 5.0) / 100
        take_profit = st.slider("Take Profit (%)", 1.0, 20.0, 10.0) / 100
        bot.ticker = ticker
        bot.years = years
        bot.leverage = leverage
        bot.stop_loss_pct = stop_loss
        bot.take_profit_pct = take_profit

    try:
        df, rec = bot.run_cycle()
        if df is None or rec is None:
            st.error("Run cycle failed. Please check logs for details.")
            return

        st.subheader("Trading Performance")
        start_date = (datetime.today() - timedelta(days=bot.years * 365)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
        fig = plot_results(df, bot.ticker, start_date, end_date)
        st.pyplot(fig)

        initial_value = bot.initial_portfolio
        final_value = df['Portfolio_Value'].iloc[-1]
        return_pct = (final_value / initial_value - 1) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Portfolio Value", f"${final_value:,.2f}")
            max_dd = (df['Portfolio_Value'].min() / df['Portfolio_Value'].max() - 1) * 100
            st.metric("Maximum Drawdown", f"{max_dd:.1f}%")
        with col2:
            st.metric("Total Return", f"{return_pct:.1f}%")
            st.metric("Volatility", f"{df['Volatility_21'].mean() * 100:.1f}%")

        st.subheader("Trade Recommendation")
        if rec['action'] == 'BUY':
            st.success(f"Action: BUY\nPrice: ${rec['current_price']:.2f}\nShares: {rec['num_shares']}\n"
                       f"Leverage: {rec['leverage']}x\nStop Loss: ${rec['stop_loss']:.2f}\nTake Profit: ${rec['take_profit']:.2f}")
        else:
            st.info(f"Action: HOLD/NO POSITION\nPrice: ${rec['current_price']:.2f}")

    except Exception as e:
        st.error(f"System Error: {e}")
        logging.error(f"Unhandled error in main: {e}")
        st.stop()

    if st.button("Clean Up Environment"):
        cleanup_temp_files()
        st.success("Cleanup complete.")

if __name__ == "__main__":
    main()
