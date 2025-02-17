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
st.set_page_config(layout="wide", page_title="Pro Trading System")
plt.style.use("ggplot")
pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings('ignore')
logging.basicConfig(filename="trading_bot.log", level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

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
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    df = df[['Close', 'Volume', 'High', 'Low']].rename(columns={'Close': 'Price'})
    full_index = pd.date_range(start=start, end=end, freq='B')
    df = df.reindex(full_index).ffill().dropna()
    df.to_csv("uniform_data.csv")
    return df

@st.cache_data
def cached_uniform_data(ticker, years):
    """Cache the uniform data so repeated calls do not re-download."""
    if os.path.exists("uniform_data.csv"):
        st.write("Loading uniform data from file...")
        df = pd.read_csv("uniform_data.csv", index_col=0, parse_dates=True)
        # Ensure uniformity
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        if not df.index.equals(full_index):
            st.write("Reindexing uniform data...")
            df = df.reindex(full_index).ffill().dropna()
        return df
    else:
        st.write("Downloading and uniformizing data...")
        return collect_uniform_data(ticker, years)

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
        df = df.copy()
        full_index = df.index
        df['SMA_50'] = df['Price'].rolling(50, min_periods=1).mean().reindex(full_index, method='ffill')
        df['SMA_200'] = df['Price'].rolling(200, min_periods=1).mean().reindex(full_index, method='ffill')
        df['RSI_14'] = self.calculate_rsi(df['Price'], 14).reindex(full_index, method='ffill')
        df['ATR_14'] = self.calculate_atr(df).reindex(full_index, method='ffill')
        df['Volume_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean().reindex(full_index, method='ffill')
        df['Daily_Return'] = df['Price'].pct_change().reindex(full_index, fill_value=0)
        df['Volatility_21'] = df['Daily_Return'].rolling(21, min_periods=1).std().reindex(full_index, fill_value=0) * np.sqrt(TRADING_DAYS_YEAR)
        return df.dropna()

    def generate_signals(self, df):
        """Generate trading signals ensuring uniform index across all Series."""
        df = df.copy()
        full_index = df.index
        price_cond = (df['Price'] > df['SMA_50']).reindex(full_index, fill_value=False)
        rsi_cond = (df['RSI_14'] > 30).reindex(full_index, fill_value=False)
        volume_cond = (df['Volume'] > df['Volume_MA_20']).reindex(full_index, fill_value=False)
        weekday_cond = df.index.weekday < 5  # Business days
        df['Signal'] = np.where(price_cond & rsi_cond & volume_cond & weekday_cond, 1, 0)
        df['Signal'] = df['Signal'].shift(1).fillna(0)
        return df

    def backtest(self, df):
        """Run a simple backtest with dynamic position sizing."""
        df = df.copy()
        df['Position'] = df['Signal'].diff()
        df['Shares'] = 0
        df['Portfolio_Value'] = self.initial_portfolio
        self.portfolio_value = self.initial_portfolio
        self.position = 0

        for i in df.index:
            current_signal = df.at[i, 'Signal']
            price = df.at[i, 'Price']
            # Close position if signal turns 0
            if self.position > 0 and current_signal == 0:
                shares = self.position
                self.portfolio_value += shares * price
                self.position = 0
                df.at[i, 'Shares'] = 0
            # Open position if signal is 1 and no current position
            elif current_signal == 1 and self.position == 0:
                shares = self.calculate_position_size(price, df.at[i, 'ATR_14'], df.at[i, 'Volatility_21'])
                if shares > 0:
                    self.position = shares
                    required_margin = (shares * price) / self.leverage
                    if required_margin > self.portfolio_value:
                        shares = int((self.portfolio_value * self.leverage) // price)
                        required_margin = (shares * price) / self.leverage
                    self.portfolio_value -= required_margin
                    df.at[i, 'Shares'] = shares
            df.at[i, 'Portfolio_Value'] = self.portfolio_value + (self.position * price)
        df['Portfolio_Value'] = df['Portfolio_Value'].ffill().fillna(self.initial_portfolio)
        return df

    def calculate_rsi(self, series, period):
        delta = series.diff().dropna()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, df):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Price'].shift())
        low_close = np.abs(df['Low'] - df['Price'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(14, min_periods=1).mean()

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
        df_raw = cached_uniform_data(self.ticker, self.years)
        df_feat = self.calculate_features(df_raw)
        df_signals = self.generate_signals(df_feat)
        df_backtest = self.backtest(df_signals)
        df_final = simulate_leveraged_cumulative_return(df_backtest, leverage=self.leverage)
        rec = self.calculate_trade_recommendation(df_final)
        self.save_results(df_final)
        return df_final, rec

# -----------------------------------------------------------------------------
# Simulation of Strategy Returns Module
# -----------------------------------------------------------------------------
def simulate_leveraged_cumulative_return(df, leverage=5):
    """
    Calculate daily returns and strategy returns by forcing uniform multiplication.
    We reset the index on daily_return and Signal to a default integer index, multiply, 
    then reassemble the result as a Series with the original DatetimeIndex.
    """
    df['daily_return'] = df['Price'].pct_change().fillna(0).astype(float)
    if 'Signal' not in df.columns:
        df['Signal'] = 0.0
    else:
        df['Signal'] = df['Signal'].astype(float)
    df['Signal'] = df['Signal'].reindex(df.index, fill_value=0)
    
    # Reset index for both Series to default integer index
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

# -----------------------------------------------------------------------------
# Plotting Function
# -----------------------------------------------------------------------------
def plot_results(df, ticker, start_date, end_date):
    """Plot Price and cumulative return."""
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

# -----------------------------------------------------------------------------
# Optional Cleanup Function
# -----------------------------------------------------------------------------
def cleanup_temp_files():
    """Delete temporary files created by the program."""
    for file in ["uniform_data.csv", "trading_results.csv"]:
        if os.path.isfile(file):
            st.write(f"Deleting temporary file: {file}")
            os.remove(file)

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
        st.subheader("Trading Performance")
        start_date = (datetime.today() - timedelta(days=bot.years * 365)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
        fig = plot_results(df, bot.ticker, start_date, end_date)
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Portfolio Value", f"${df['Portfolio_Value'].iloc[-1]:,.2f}")
            max_dd = (df['Portfolio_Value'].min() / df['Portfolio_Value'].max() - 1) * 100
            st.metric("Maximum Drawdown", f"{max_dd:.1f}%")
        with col2:
            total_return = (df['Portfolio_Value'].iloc[-1] / bot.initial_portfolio - 1) * 100
            st.metric("Total Return", f"{total_return:.1f}%")
            st.metric("Volatility", f"{df['Volatility_21'].mean() * 100:.1f}%")
        
        st.subheader("Trade Recommendation")
        if rec['action'] == 'BUY':
            st.success(f"Action: BUY\nPrice: ${rec['current_price']:.2f}\nShares: {rec['num_shares']}\n"
                       f"Leverage: {rec['leverage']}x\nStop Loss: ${rec['stop_loss']:.2f}\nTake Profit: ${rec['take_profit']:.2f}")
        else:
            st.info(f"Action: HOLD/NO POSITION\nPrice: ${rec['current_price']:.2f}")
        
    except Exception as e:
        st.error(f"System Error: {str(e)}")
        st.stop()

    if st.button("Clean Up Environment"):
        cleanup_temp_files()
        st.success("Cleanup complete.")

if __name__ == "__main__":
    main()
