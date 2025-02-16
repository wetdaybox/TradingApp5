# app.py - Final Integrated Version
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import unittest

# Configure app for mobile
st.set_page_config(page_title="Trading System", layout="wide")
plt.style.use('seaborn-darkgrid')  # High-contrast theme

# ---------------------
# Core Functions
# ---------------------
@st.cache_data
def get_data(ticker='AAPL', days=1095):
    """Get and clean data with error handling"""
    try:
        end = datetime.today()
        start = end - timedelta(days=days)
        df = yf.download(ticker, start=start, end=end, progress=False)
        df = df[['Close']].rename(columns={'Close':'Price'})
        df.index = pd.to_datetime(df.index)
        df = df.resample('B').last().ffill()
        return df.dropna()
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def calculate_strategy(df, sma_window=50, risk_pct=0.05, reward_ratio=2):
    """Verified strategy calculations"""
    df['SMA'] = df['Price'].rolling(sma_window).mean()
    
    # Signal logic with validation
    signals = (df['Price'] > df['SMA']).astype(int)
    df['Signal'] = signals.shift(1).fillna(0)  # Prevent look-ahead bias
    
    # Entry/exit prices
    df['Entry_Price'] = np.where(df['Signal'].diff() == 1, df['Price'], np.nan)
    df['Stop_Loss'] = df['Entry_Price'] * (1 - risk_pct)
    df['Take_Profit'] = df['Entry_Price'] * (1 + (risk_pct * reward_ratio))
    
    return df.ffill()

# ---------------------
# Mobile Interface
# ---------------------
with st.sidebar:
    st.header("⚙️ Controls")
    ticker = st.text_input("Stock", "AAPL").upper()
    years = st.slider("Years History 📅", 1, 5, 3)
    risk = st.slider("Risk % ⚠️", 1.0, 10.0, 5.0) / 100
    reward = st.selectbox("Reward Ratio 🎯", [2, 3, 4], index=0)

df = get_data(ticker, days=years*365)

if not df.empty:
    df = calculate_strategy(df, risk_pct=risk, reward_ratio=reward)
    current_signal = df['Signal'].iloc[-1]
    last_trade = df[df['Entry_Price'].notna()].iloc[-1] if current_signal else None
    
    # Mobile-optimized layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("📊 Current Status")
        st.metric("Price 💵", f"${df['Price'].iloc[-1]:.2f}")
        st.metric("50-day SMA 📈", f"${df['SMA'].iloc[-1]:.2f}")
        
    with col2:
        if current_signal:
            st.subheader("✅ Active Trade")
            st.metric("Entry Price 🟢", f"${last_trade['Entry_Price']:.2f}")
            st.metric("Stop Loss 🔴", 
                     f"${last_trade['Stop_Loss']:.2f}", 
                     delta=f"-{risk*100:.0f}%")
            st.metric("Take Profit 🟩", 
                     f"${last_trade['Take_Profit']:.2f}", 
                     delta=f"+{risk*100*reward:.0f}%")
        else:
            st.subheader("🛑 No Position")
            st.metric("Next Signal 🔄", 
                      "Price > 50-day SMA", 
                      help="Waiting for entry condition")

    # Enhanced mobile plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Price'], label='Price', lw=2, color='#1f77b4')
    ax.plot(df.index, df['SMA'], label='50-day SMA', ls='--', color='#ff7f0e')
    
    if current_signal:
        ax.axhline(last_trade['Stop_Loss'], color='#d62728', lw=2.5, 
                  label=f'Stop Loss (${last_trade["Stop_Loss"]:.2f})')
        ax.axhline(last_trade['Take_Profit'], color='#2ca02c', lw=2.5,
                  label=f'Take Profit (${last_trade["Take_Profit"]:.2f})')
        
        # Mobile-friendly annotations
        bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="0.5", alpha=0.9)
        ax.annotate(f'STOP\n${last_trade["Stop_Loss"]:.2f}',
                   xy=(df.index[-1], last_trade['Stop_Loss']),
                   xytext=(-10, 0), textcoords='offset points',
                   ha='right', va='center', color='#d62728',
                   fontsize=12, bbox=bbox_props)
        ax.annotate(f'TAKE PROFIT\n${last_trade["Take_Profit"]:.2f}',
                   xy=(df.index[-1], last_trade['Take_Profit']),
                   xytext=(-10, 0), textcoords='offset points',
                   ha='right', va='center', color='#2ca02c',
                   fontsize=12, bbox=bbox_props)
    
    ax.set_title(f"{ticker} Trading Plan", fontsize=16, pad=20)
    ax.legend(loc='upper left', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(fig)

# ---------------------
# Validation Tests (Hidden)
# ---------------------
class TestTradingSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        prices = np.concatenate([np.linspace(90, 110, 50), np.linspace(115, 95, 50)])
        cls.df = pd.DataFrame({'Price': prices}, index=dates)
        cls.df = calculate_strategy(cls.df)

    def test_signal_accuracy(self):
        expected_signals = np.zeros(100)
        expected_signals[50:] = 1  # Price crosses SMA at midpoint
        expected_signals = np.roll(expected_signals, 1)  # Shift for look-ahead
        expected_signals[0] = 0
        np.testing.assert_array_equal(self.df['Signal'].values, expected_signals)

    def test_entry_prices(self):
        entries = self.df[self.df['Signal'].diff() == 1]
        self.assertEqual(len(entries), 1, "Should have one entry at crossover")
        self.assertAlmostEqual(entries['Entry_Price'].iloc[0], 110.0, delta=0.1)

if st.secrets.get("run_tests", False):
    with st.expander("🔍 Validation Tests", expanded=False):
        unittest.main(argv=[''], verbosity=2, exit=False)

# ---------------------
# Footer
# ---------------------
st.markdown("---")
st.markdown("""
**📚 Trading Rules**  
- Enter long when closing price > 50-day SMA  
- Position size: (Account risk) / (Entry price × Risk %)  
- Exit at stop loss or take profit levels  

*⚠️ Disclaimer: Educational use only. Past performance ≠ future results.*  
""")
