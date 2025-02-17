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
        self.initial_portfolio = portfolio_value
        self.portfolio_value = portfolio_value  # Tracks cash available
        self.leverage = leverage
        self.sma_window = sma_window
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.years = years

    @st.cache_data(ttl=3600, allow_output_mutation=True)
    def fetch_data(self, ticker, years):
        """Fetch and align data with a complete business-day index."""
        end = datetime.today()
        start = end - timedelta(days=years * 365)
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                st.error(f"No data found for ticker: {ticker}")
                return None
            df = df[['Close', 'Volume', 'High', 'Low']].rename(columns={'Close': 'Price'})
            full_index = pd.date_range(start=start, end=end, freq='B')
            df = df.reindex(full_index).ffill().dropna()
            return df
        except Exception as e:
            st.error(f"Failed to fetch data: {str(e)}")
            return None

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
        """Run backtest with proper position sizing and leverage handling."""
        df = df.copy()
        df['Position'] = df['Signal'].diff().fillna(0)
        df['Shares'] = 0.0
        cash = self.initial_portfolio
        shares_held = 0.0
        portfolio_values = []

        for i, row in df.iterrows():
            current_price = row['Price']
            
            # Update portfolio value for current day before any trades
            current_portfolio = cash + shares_held * current_price
            portfolio_values.append(current_portfolio)
            
            # Execute trades if there's a signal
            if df.at[i, 'Position'] != 0:
                if df.at[i, 'Position'] == 1:  # Buy signal
                    # Calculate position size with leverage
                    risk_per_share = current_price * self.stop_loss_pct
                    position_size = (cash * self.leverage * 0.01) / risk_per_share  # Using 1% risk per trade
                    max_shares = (cash * self.leverage) // current_price  # Max shares with leverage
                    shares = min(position_size, max_shares)
                    
                    # Ensure we don't exceed available margin
                    cost = shares * current_price
                    margin_required = cost / self.leverage
                    if margin_required > cash:
                        shares = (cash * self.leverage) // current_price
                        cost = shares * current_price
                        margin_required = cost / self.leverage
                    
                    # Update cash and shares
                    cash -= margin_required
                    shares_held += shares
                elif df.at[i, 'Position'] == -1:  # Sell signal
                    # Sell all shares and add proceeds to cash
                    proceeds = shares_held * current_price
                    cash += proceeds
                    shares_held = 0.0
                
                # Update shares in DataFrame
                df.at[i, 'Shares'] = shares_held
            
            # Update portfolio value post-trade
            df.at[i, 'Portfolio_Value'] = cash + shares_held * current_price

        # Fill portfolio values
        df['Portfolio_Value'] = portfolio_values
        return df

    def calculate_rsi(self, series, period=14):
        """Calculate RSI with proper NaN handling."""
        delta = series.diff(1).dropna()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, df):
        """Calculate ATR."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Price'].shift())
        low_close = np.abs(df['Low'] - df['Price'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(14, min_periods=1).mean()

    def run_cycle(self):
        """Run one trading cycle."""
        df = self.fetch_data(self.ticker, self.years)
        if df is None:
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
        leverage = st.slider("Leverage", 1, 10, 5)
        bot.ticker = ticker
        bot.years = years
        bot.leverage = leverage

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

    initial_value = bot.initial_portfolio
    final_value = df['Portfolio_Value'].iloc[-1]
    return_pct = (final_value / initial_value - 1) * 100

    col1, col2 = st.columns(2)
    col1.metric("Final Portfolio Value", f"${final_value:,.2f}")
    col2.metric("Total Return", f"{return_pct:.1f}%")

    st.subheader("Portfolio Value Over Time")
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(df.index, df['Portfolio_Value'], label='Portfolio Value', color='green')
    ax2.grid(True)
    st.pyplot(fig2)

if __name__ == "__main__":
    main()
