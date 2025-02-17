import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(layout="wide", page_title="Pro Trading System")
plt.style.use("ggplot")
pd.set_option('mode.chained_assignment', None)
logging.basicConfig(filename="trading_bot.log", level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(message)s")

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

    @st.cache_data
    def fetch_data(_self, ticker, years):
        """Fetch and align data with a complete business day index."""
        end = datetime.today()
        start = end - timedelta(days=years * 365)
        
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                raise ValueError(f"No data found for ticker: {ticker}")
            df = df[['Close', 'Volume', 'High', 'Low']].rename(columns={'Close': 'Price'})
            full_index = pd.date_range(start=start, end=end, freq='B')
            df = df.reindex(full_index).ffill().dropna()
            return df
        except Exception as e:
            st.error(f"Failed to fetch data: {str(e)}")
            st.stop()

    def calculate_features(self, df):
        """Calculate technical indicators and force uniform index."""
        df = df.copy()
        full_index = df.index  # Assumed complete from fetch_data
        df['SMA_50'] = df['Price'].rolling(50, min_periods=1).mean().reindex(full_index, method='ffill')
        df['SMA_200'] = df['Price'].rolling(200, min_periods=1).mean().reindex(full_index, method='ffill')
        df['RSI_14'] = self.calculate_rsi(df['Price'], 14).reindex(full_index, method='ffill')
        df['ATR_14'] = self.calculate_atr(df).reindex(full_index, method='ffill')
        df['Volume_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean().reindex(full_index, method='ffill')
        df['Daily_Return'] = df['Price'].pct_change().reindex(full_index, fill_value=0)
        df['Volatility_21'] = df['Daily_Return'].rolling(21, min_periods=1).std().reindex(full_index, fill_value=0) * np.sqrt(TRADING_DAYS_YEAR)
        return df.dropna()

    def generate_signals(self, df):
        """Generate trading signals with explicit uniform alignment."""
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
                pos = df.index.get_loc(i)
                prev_shares = df.iloc[pos - 1]['Shares'] if pos > 0 else 0
                self.portfolio_value += prev_shares * row['Price']
                df.at[i, 'Shares'] = 0
            
            df.at[i, 'Portfolio_Value'] = self.portfolio_value + (df.at[i, 'Shares'] * row['Price'])
        
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
# Uniform Data Preparation (Cached)
# ======================
@st.cache_data
def prepare_uniform_data_cached(ticker, years):
    """Fetch uniform data so repeated calls do not refetch."""
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        st.error(f"No data found for ticker: {ticker}")
        st.stop()
    df = df[['Close', 'Volume', 'High', 'Low']].rename(columns={'Close': 'Price'})
    full_index = pd.date_range(start=start, end=end, freq='B')
    df = df.reindex(full_index).ffill().dropna()
    return df

# ======================
# Simulation of Strategy Returns
# ======================
def simulate_leveraged_cumulative_return(df, leverage=5):
    """
    Calculate daily returns, force daily_return and Signal to be one-dimensional Series,
    explicitly align them on the index, and multiply elementwise.
    """
    df['daily_return'] = df['Price'].pct_change().fillna(0).astype(float)
    if 'Signal' not in df.columns:
        df['Signal'] = 0.0
    else:
        df['Signal'] = df['Signal'].astype(float)
    df['Signal'] = df['Signal'].reindex(df.index, fill_value=0)
    
    # Force one-dimensional Series using .squeeze() and construct new Series from numpy arrays.
    dr = pd.Series(df['daily_return'].values, index=df.index)
    sig = pd.Series(df['Signal'].values, index=df.index)
    
    # Explicitly align the two Series
    dr_aligned, sig_aligned = dr.align(sig, axis=0, fill_value=0)
    
    # Debug output
    st.write("Debug - daily_return type:", type(dr_aligned), "shape:", dr_aligned.shape)
    st.write("Debug - Signal type:", type(sig_aligned), "shape:", sig_aligned.shape)
    st.write("Debug - daily_return head:", dr_aligned.head())
    st.write("Debug - Signal head:", sig_aligned.head())
    logging.info(f"Aligned daily_return shape: {dr_aligned.shape}, head: {dr_aligned.head()}")
    logging.info(f"Aligned Signal shape: {sig_aligned.shape}, head: {sig_aligned.head()}")
    
    multiplied = dr_aligned * sig_aligned
    df['strategy_return'] = leverage * multiplied
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    return df

def save_results(df, filename="trading_results.csv"):
    """Save simulation results to a CSV file."""
    result = pd.DataFrame({
        "timestamp": [datetime.now()],
        "current_price": [df['Price'].iloc[-1]],
        "cumulative_return": [df['cumulative_return'].iloc[-1]]
    })
    if os.path.isfile(filename):
        result.to_csv(filename, mode="a", header=False, index=False)
    else:
        result.to_csv(filename, index=False)
    logging.info("Results saved to " + filename)

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

# ======================
# Trading Bot Class
# ======================
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

    def calculate_features(self, df):
        """Calculate technical indicators and force uniform index."""
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
        """Generate trading signals with explicit uniform alignment."""
        df = df.copy()
        full_index = df.index
        price_cond = (df['Price'] > df['SMA_50']).reindex(full_index, fill_value=False)
        rsi_cond = (df['RSI_14'] > 30).reindex(full_index, fill_value=False)
        volume_cond = (df['Volume'] > df['Volume_MA_20']).reindex(full_index, fill_value=False)
        weekday_cond = df.index.weekday < 5
        df['Signal'] = np.where(price_cond & rsi_cond & volume_cond & weekday_cond, 1, 0)
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
                pos = df.index.get_loc(i)
                prev_shares = df.iloc[pos - 1]['Shares'] if pos > 0 else 0
                self.portfolio_value += prev_shares * row['Price']
                df.at[i, 'Shares'] = 0
            df.at[i, 'Portfolio_Value'] = self.portfolio_value + (df.at[i, 'Shares'] * row['Price'])
        
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

    def run_cycle(self):
        """
        Run one trading cycle:
          1. Define date range.
          2. Get uniform data from cache.
          3. Calculate technical features.
          4. Generate signals.
          5. Backtest the strategy.
          6. Simulate cumulative returns.
          7. Calculate trade recommendation.
          8. Save results.
        """
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=self.years * 365)).strftime('%Y-%m-%d')
        
        df_raw = prepare_uniform_data_cached(self.ticker, self.years)
        df_feat = self.calculate_features(df_raw)
        df_signals = self.generate_signals(df_feat)
        df_backtest = self.backtest(df_signals)
        df_final = simulate_leveraged_cumulative_return(df_backtest, leverage=self.leverage)
        recommendation = calculate_trade_recommendation(df_final, self.portfolio_value, self.leverage,
                                                         self.stop_loss_pct, self.take_profit_pct)
        save_results(df_final)
        return df_final, recommendation, start_date, end_date

# ======================
# Optional Cleanup Functions
# ======================
def cleanup_temp_files():
    """Delete temporary files created by the program."""
    for file in ["uniform_data.csv", "trading_results.csv"]:
        if os.path.isfile(file):
            print(f"Deleting temporary file: {file}")
            os.remove(file)

# ======================
# Streamlit Interface
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
        
    try:
        df, rec, start_date, end_date = bot.run_cycle()
        st.subheader("Trading Performance")
        fig = plot_results(df, bot.ticker, start_date, end_date)
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Portfolio Value", f"${df['Portfolio_Value'].iloc[-1]:,.2f}")
            max_dd = (df['Portfolio_Value'].min() / df['Portfolio_Value'].max() - 1) * 100
            st.metric("Maximum Drawdown", f"{max_dd:.1f}%")
        with col2:
            total_return = (df['Portfolio_Value'].iloc[-1] / 100000 - 1) * 100
            st.metric("Total Return", f"{total_return:.1f}%")
            st.metric("Volatility", f"{df['Volatility_21'].mean() * 100:.1f}%")
        
        st.subheader("Trade Recommendation")
        if rec['action'] == 'BUY':
            st.success(f"Action: BUY\nPrice: ${rec['current_price']:.2f}\nShares: {rec['num_shares']}\n"
                       f"Leverage: {rec['leverage']}x\nStop-Loss: ${rec['stop_loss']:.2f}\nTake-Profit: ${rec['take_profit']:.2f}")
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
