import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pytz
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
REFRESH_INTERVAL = 60  # seconds

# Initialize session state variables
if 'manual_price' not in st.session_state:
    st.session_state.manual_price = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")

# -------------------------------
# Indicator Calculation Functions
# -------------------------------
def get_rsi_wilder(data, window=14):
    """Calculate RSI using Wilder's smoothing method."""
    if len(data) < window + 1:
        return pd.Series([np.nan] * len(data), index=data.index)
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean().iloc[window-1]
    avg_loss = loss.rolling(window=window, min_periods=window).mean().iloc[window-1]
    rsi_series = [np.nan] * (window - 1)
    rsi_series.append(100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss != 0 else 100)
    for i in range(window, len(data)):
        current_gain = gain.iloc[i]
        current_loss = loss.iloc[i]
        avg_gain = (avg_gain * (window - 1) + current_gain) / window
        avg_loss = (avg_loss * (window - 1) + current_loss) / window
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi_series.append(100 - (100 / (1 + rs)) if avg_loss != 0 else 100)
    return pd.Series(rsi_series, index=data.index)

def get_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line and histogram."""
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def get_bollinger_bands(data, window=20, num_std=2):
    """Calculate SMA and Bollinger Bands."""
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

# -------------------------------
# Data Fetching Functions with Caching
# -------------------------------
@st.cache_data(ttl=600)
def fetch_fx_rate():
    """Fetch FX rate with a TTL of 10 minutes."""
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='5m', progress=False)
        if not fx_data.empty:
            return fx_data['Close'].iloc[-1]
        else:
            return 0.80
    except Exception as e:
        st.error(f"FX error: {str(e)}")
        return 0.80

@st.cache_data(ttl=30)
def fetch_realtime_data(pair):
    """Fetch 48 hours of 5-minute interval data and compute technical indicators."""
    try:
        data = yf.download(pair, period='2d', interval='5m', progress=False)
        if not data.empty:
            data['RSI'] = get_rsi_wilder(data)
            macd_line, signal_line, histogram = get_macd(data)
            data['MACD'] = macd_line
            data['Signal'] = signal_line
            sma, upper_band, lower_band = get_bollinger_bands(data)
            data['SMA'] = sma
            data['UpperBB'] = upper_band
            data['LowerBB'] = lower_band
            st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        return data
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def get_price_data(pair):
    """Return the latest price (converted by FX rate) or manual override."""
    data = fetch_realtime_data(pair)
    fx_rate = fetch_fx_rate()
    if st.session_state.manual_price is not None:
        return st.session_state.manual_price, True
    if not data.empty:
        return data['Close'].iloc[-1] / fx_rate, False
    return None, False

def calculate_levels(pair, current_price, tp_percent, sl_percent):
    """Calculate levels based on a 24h data window."""
    data = fetch_realtime_data(pair)
    if data.empty or len(data) < 288:
        return None
    try:
        full_day_data = data.iloc[-288:]
        recent_low = full_day_data['Low'].min()
        recent_high = full_day_data['High'].max()
        fx_rate = fetch_fx_rate()
        last_rsi = data['RSI'].iloc[-1]
        # ATR (14-period) calculation
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        return {
            'buy_zone': round(recent_low * 0.98 / fx_rate, 2),
            'take_profit': round(current_price * (1 + tp_percent / 100), 2),
            'stop_loss': round(current_price * (1 - sl_percent / 100), 2),
            'rsi': round(last_rsi, 1),
            'high': round(recent_high / fx_rate, 2),
            'low': round(recent_low / fx_rate, 2),
            'volatility': round(atr / fx_rate, 2)
        }
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return None

# -------------------------------
# Simple Backtesting Function
# -------------------------------
def backtest_strategy(pair, tp_percent, sl_percent, initial_capital=1000):
    """Run a simple backtest based on RSI signals."""
    data = fetch_realtime_data(pair)
    if data.empty:
        return None
    fx_rate = fetch_fx_rate()
    data = data.copy()
    data['Price'] = data['Close'] / fx_rate
    # Signal: buy if RSI is oversold, sell if overbought
    data['Signal'] = 0
    data.loc[data['RSI'] < RSI_OVERSOLD, 'Signal'] = 1
    data.loc[data['RSI'] > RSI_OVERBOUGHT, 'Signal'] = -1
    
    position = 0
    cash = initial_capital
    portfolio_values = []
    for i in range(1, len(data)):
        # Buy if signal is 1 and not in position
        if data['Signal'].iloc[i] == 1 and position == 0:
            position = cash / data['Price'].iloc[i]
            cash = 0
        # Sell if signal is -1 and in position
        elif data['Signal'].iloc[i] == -1 and position > 0:
            cash = position * data['Price'].iloc[i]
            position = 0
        portfolio_value = cash + position * data['Price'].iloc[i]
        portfolio_values.append(portfolio_value)
    data = data.iloc[1:].copy()
    data['Portfolio'] = portfolio_values
    total_return = (portfolio_values[-1] - initial_capital) / initial_capital * 100
    return data, total_return

# -------------------------------
# Main Application Function
# -------------------------------
def main():
    st.set_page_config(page_title="Crypto Trader Pro", layout="centered")
    st.title("📈 Real-Time Crypto Assistant")
    st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="main_refresh")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Asset:", CRYPTO_PAIRS)
        use_manual = st.checkbox("Enter Price Manually")
        if use_manual:
            st.session_state.manual_price = st.number_input(
                "Manual Price (£)", min_value=0.01,
                value=st.session_state.manual_price or 1000.0
            )
        else:
            st.session_state.manual_price = None

        account_size = st.number_input("Portfolio Value (£)", min_value=100.0, value=1000.0, step=100.0)
        risk_profile = st.select_slider("Risk Profile:", options=['Safety First', 'Balanced', 'High Risk'])
        risk_reward = st.slider("Risk/Reward Ratio", 1.0, 5.0, 3.0, 0.5)
        tp_percent = st.slider("Take Profit %", 1.0, 30.0, 15.0)
        sl_percent = st.slider("Stop Loss %", 1.0, 10.0, 5.0)
        backtest_button = st.button("Run Backtest")
    
    with col2:
        update_diff = (datetime.now() - datetime.strptime(st.session_state.last_update, "%H:%M:%S")).seconds
        recency_color = "green" if update_diff < 120 else "orange" if update_diff < 300 else "red"
        st.markdown(f"🕒 Last update: <span style='color:{recency_color}'>{st.session_state.last_update}</span>",
                    unsafe_allow_html=True)
        
        current_price, is_manual = get_price_data(pair)
        if current_price:
            levels = calculate_levels(pair, current_price, tp_percent, sl_percent)
            if levels:
                buy_signal = levels['rsi'] < RSI_OVERSOLD
                take_profit_signal = levels['rsi'] > RSI_OVERBOUGHT
                
                alert_cols = st.columns(3)
                rsi_color = "green" if levels['rsi'] < RSI_OVERSOLD else "red" if levels['rsi'] > RSI_OVERBOUGHT else "gray"
                alert_cols[0].markdown(f"<span style='color:{rsi_color};font-size:24px'>{levels['rsi']}</span>",
                                       unsafe_allow_html=True)
                alert_cols[0].caption("RSI (Oversold <30, Overbought >70)")
                alert_cols[1].metric("24h Range", f"£{levels['low']:,.2f}-£{levels['high']:,.2f}")
                alert_cols[2].metric("Volatility", f"£{levels['volatility']:,.2f}")
                
                with st.expander("Trading Strategy"):
                    st.write(f"""
                    **Recommended Action:**  
                    {'🔥 Consider buying - Oversold market' if buy_signal else 
                     '💰 Consider profit taking - Overbought market' if take_profit_signal else 
                     '⏳ Hold - Neutral market conditions'}
                    
                    **Entry Zone:** £{levels['buy_zone']:,.2f}  
                    **Profit Target:** £{levels['take_profit']:,.2f} (+{tp_percent}%)  
                    **Stop Loss:** £{levels['stop_loss']:,.2f} (-{sl_percent}%)
                    """)
                    
                    fig = go.Figure()
                    hist_data = fetch_realtime_data(pair)
                    fx_rate = fetch_fx_rate()
                    # Price History Trace
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['Close'] / fx_rate,
                        name='Price History',
                        line=dict(color='#1f77b4')
                    ))
                    # MACD Trace on secondary y-axis
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['MACD'],
                        name='MACD',
                        line=dict(color='purple'),
                        yaxis='y2'
                    ))
                    # Bollinger Bands: Upper, SMA, Lower
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['UpperBB'] / fx_rate,
                        name='Upper BB',
                        line=dict(color='gray', dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['SMA'] / fx_rate,
                        name='SMA',
                        line=dict(color='black')
                    ))
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['LowerBB'] / fx_rate,
                        name='Lower BB',
                        line=dict(color='gray', dash='dash')
                    ))
                    # Horizontal Levels
                    fig.add_hline(y=levels['buy_zone'], line_dash="dot", annotation_text="Buy Zone", line_color="green")
                    fig.add_hline(y=levels['take_profit'], line_dash="dot", annotation_text="Profit Target", line_color="blue")
                    fig.add_hline(y=levels['stop_loss'], line_dash="dot", annotation_text="Stop Loss", line_color="red")
                    
                    # RSI-based Buy/Sell Signals
                    signals = pd.DataFrame(index=hist_data.index)
                    signals['Buy'] = hist_data['RSI'] < RSI_OVERSOLD
                    signals['Sell'] = hist_data['RSI'] > RSI_OVERBOUGHT
                    fig.add_trace(go.Scatter(
                        x=signals[signals['Buy']].index,
                        y=(hist_data.loc[signals['Buy'], 'Close'] / fx_rate),
                        mode='markers',
                        name='Buy Signals',
                        marker=dict(color='green', size=8, symbol='triangle-up')
                    ))
                    fig.add_trace(go.Scatter(
                        x=signals[signals['Sell']].index,
                        y=(hist_data.loc[signals['Sell'], 'Close'] / fx_rate),
                        mode='markers',
                        name='Sell Signals',
                        marker=dict(color='red', size=8, symbol='triangle-down')
                    ))
                    
                    # Layout with secondary y-axis for MACD
                    fig.update_layout(
                        yaxis=dict(title="Price (£)"),
                        yaxis2=dict(title="MACD", overlaying='y', side='right', showgrid=False),
                        xaxis_title="Time",
                        title=f"Price Chart with Technical Indicators for {pair}",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.write("## Position Builder")
                risk_amount = st.slider("Risk Amount (£)", 10.0, account_size, 100.0)
                # Position size based on the distance from current price to stop loss
                position_size = risk_amount / abs(current_price - levels['stop_loss'])
                st.write(f"""
                **Suggested Position:**  
                - Size: {position_size:.4f} {pair.split('-')[0]}  
                - Value: £{(position_size * current_price):,.2f}  
                - Risk/Reward: 1:{risk_reward}
                """)
            else:
                st.error("Market data unavailable")
        else:
            st.warning("Waiting for price data...")
        
        # Backtesting Section
        if backtest_button:
            backtest_result = backtest_strategy(pair, tp_percent, sl_percent, initial_capital=account_size)
            if backtest_result is not None:
                bt_data, total_return = backtest_result
                st.subheader("Backtest Results")
                st.line_chart(bt_data['Portfolio'])
                st.write(f"**Total Return:** {total_return:.2f}%")
            else:
                st.error("Backtest could not be run due to insufficient data.")

if __name__ == "__main__":
    main()
