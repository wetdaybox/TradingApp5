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
        self.portfolio_value = portfolio_value  # Tracks available equity
        self.leverage = leverage
        self.sma_window = sma_window
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.years = years
        self.position = 0  # Current position in shares
        self.margin_used = 0  # Tracks margin committed to open positions

    @st.cache_data
    def fetch_data(self, ticker, years):
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
        df = df.copy()
        price_cond = df['Price'] > df['SMA_50']
        rsi_cond = df['RSI_14'] > 30
        volume_cond = df['Volume'] > df['Volume_MA_20']
        weekday_cond = df.index.weekday < 5
        df['Signal'] = np.where(price_cond & rsi_cond & volume_cond & weekday_cond, 1, 0)
        df['Signal'] = df['Signal'].shift(1).fillna(0)
        return df

    def backtest(self, df):
        df = df.copy()
        df['Position'] = df['Signal'].diff()
        df['Shares'] = 0
        df['Portfolio_Value'] = self.initial_portfolio
        self.portfolio_value = self.initial_portfolio
        self.margin_used = 0
        self.position = 0

        for i in df.index:
            current_signal = df.at[i, 'Signal']
            price = df.at[i, 'Price']
            atr = df.at[i, 'ATR_14']
            volatility = df.at[i, 'Volatility_21']

            # Close existing position on sell signal or stop loss/take profit
            if self.position > 0 and (current_signal == 0 or self.check_exit_conditions(price)):
                self.close_position(df, i, price)

            # Open new position on buy signal
            if current_signal == 1 and self.position == 0:
                self.open_position(df, i, price, atr, volatility)

            # Update portfolio value (equity = available cash + position value - margin used)
            df.at[i, 'Portfolio_Value'] = self.portfolio_value + (self.position * price) - self.margin_used

        return df

    def open_position(self, df, index, price, atr, volatility):
        max_investment = self.portfolio_value * self.leverage
        risk_per_share = price * 0.01
        position_size = (self.portfolio_value * 0.01) / risk_per_share
        position_size *= self.leverage  # Apply leverage
        volatility_adj = 1 / (1 + volatility)
        shares = int(position_size * volatility_adj)
        
        if shares == 0:
            return
        
        required_margin = (shares * price) / self.leverage
        if required_margin > self.portfolio_value:
            shares = int((self.portfolio_value * self.leverage) // price)
            required_margin = (shares * price) / self.leverage
        
        if shares > 0:
            self.position = shares
            self.margin_used += required_margin
            self.portfolio_value -= required_margin
            df.at[index, 'Shares'] = shares

    def close_position(self, df, index, price):
        proceeds = self.position * price
        margin_return = self.margin_used
        self.portfolio_value += proceeds + margin_return
        self.margin_used = 0
        self.position = 0
        df.at[index, 'Shares'] = 0

    def check_exit_conditions(self, current_price):
        if self.position == 0:
            return False
        entry_price = self.margin_used * self.leverage / self.position
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        take_profit = entry_price * (1 + self.take_profit_pct)
        return current_price <= stop_loss or current_price >= take_profit

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

    def run_cycle(self):
        df = self.fetch_data(self.ticker, self.years)
        df = self.calculate_features(df)
        df = self.generate_signals(df)
        df = self.backtest(df)
        recommendation = self.calculate_trade_recommendation(df)
        self.save_results(df)
        return df, recommendation

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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        ax1.plot(df.index, df['Price'], label='Price', color='black')
        ax1.plot(df.index, df['SMA_50'], label='50-day SMA', color='blue', linestyle='--')
        ax1.set_title(f"{bot.ticker} Price and 50-day SMA")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(df.index, df['Portfolio_Value'], label='Portfolio Value', color='green')
        ax2.set_title("Portfolio Value Over Time")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Portfolio Value", f"${df['Portfolio_Value'].iloc[-1]:,.2f}")
            max_dd = (df['Portfolio_Value'].min() / df['Portfolio_Value'].max() - 1) * 100
            st.metric("Maximum Drawdown", f"{max_dd:.1f}%")
        with col2:
            total_return = (df['Portfolio_Value'].iloc[-1] / bot.initial_portfolio - 1) * 100
            st.metric("Total Return", f"{total_return:.1f}%")
            st.metric("Annualized Volatility", f"{df['Volatility_21'].mean() * 100:.1f}%")
        
        st.subheader("Trade Recommendation")
        if rec['action'] == 'BUY':
            st.success(f"Action: {rec['action']}\nPrice: ${rec['current_price']:.2f}\n"
                       f"Shares: {rec['num_shares']}\nLeverage: {rec['leverage']}x\n"
                       f"Stop Loss: ${rec['stop_loss']:.2f}\nTake Profit: ${rec['take_profit']:.2f}")
        else:
            st.info(f"Action: {rec['action']}\nCurrent Price: ${rec['current_price']:.2f}")
        
    except Exception as e:
        st.error(f"System Error: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()
