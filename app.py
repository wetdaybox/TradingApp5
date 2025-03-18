import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
REFRESH_INTERVAL = 60  # Seconds between auto-refreshes

# Initialize session state
if 'manual_price' not in st.session_state:
    st.session_state.manual_price = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")

# -------------------------------
# Original Indicator Calculation Function
# -------------------------------
def get_rsi(data, window=14):
    if len(data) < window + 1:
        return pd.Series([None] * len(data), index=data.index)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -------------------------------
# Data Fetching Functions (Original Method)
# -------------------------------
@st.cache_data(ttl=30)
def get_realtime_data(pair):
    try:
        data = yf.download(pair, period='2d', interval='5m', progress=False)
        if not data.empty:
            data['RSI'] = get_rsi(data)
            st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        return data
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_fx_rate():
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='5m', progress=False)
        return fx_data['Close'].iloc[-1].item() if not fx_data.empty else 0.80
    except Exception as e:
        st.error(f"FX error: {str(e)}")
        return 0.80

def get_price_data(pair):
    data = get_realtime_data(pair)
    fx_rate = get_fx_rate()
    if st.session_state.manual_price is not None:
        return st.session_state.manual_price, True
    if not data.empty and 'Close' in data.columns:
        return data['Close'].iloc[-1].item() / fx_rate, False
    return None, False

# -------------------------------
# Additional Cross-Reference Price Function
# -------------------------------
def cross_reference_price(pair):
    """Cross-check the current price using the yf.Ticker approach with a 1-day, 1-minute interval."""
    try:
        ticker = yf.Ticker(pair)
        alt_data = ticker.history(period='1d', interval='1m', progress=False)
        if not alt_data.empty:
            return alt_data['Close'].iloc[-1].item()
        else:
            return None
    except Exception as e:
        st.error(f"Alternative data error: {str(e)}")
        return None

# -------------------------------
# Levels & Backtesting (Original Logic)
# -------------------------------
def calculate_levels(pair, current_price, tp_percent, sl_percent):
    data = get_realtime_data(pair)
    if data.empty or len(data) < 288:
        return None
    try:
        full_day_data = data.iloc[-288:]
        recent_low = full_day_data['Low'].min().item()
        recent_high = full_day_data['High'].max().item()
        fx_rate = get_fx_rate()
        last_rsi = data['RSI'].iloc[-1]
        
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

def backtest_strategy(pair, tp_percent, sl_percent, initial_capital=1000):
    data = get_realtime_data(pair)
    if data.empty:
        return None
    fx_rate = get_fx_rate()
    data = data.copy()
    data['Price'] = data['Close'] / fx_rate
    data['Signal'] = 0
    data.loc[data['RSI'] < RSI_OVERSOLD, 'Signal'] = 1
    data.loc[data['RSI'] > RSI_OVERBOUGHT, 'Signal'] = -1
    
    position = 0
    cash = initial_capital
    portfolio_values = []
    for i in range(1, len(data)):
        if data['Signal'].iloc[i] == 1 and position == 0:
            position = cash / data['Price'].iloc[i]
            cash = 0
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
# Main Application
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
            st.session_state.manual_price = st.number_input("Manual Price (£)", min_value=0.01,
                                                              value=st.session_state.manual_price or 1000.0)
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
        st.markdown(f"🕒 Last update: <span style='color:{recency_color}'>{st.session_state.last_update}</span>", unsafe_allow_html=True)
        
        current_price, is_manual = get_price_data(pair)
        alt_price = cross_reference_price(pair)
        if current_price and alt_price:
            # Compute the percentage difference between the two data sources.
            diff_pct = abs(current_price - alt_price) / current_price * 100
            st.metric("Price Diff (%)", f"{diff_pct:.2f}%")
            # Optionally, compute an aggregated average price.
            aggregated_price = (current_price + alt_price) / 2
            st.write(f"Aggregated Price: £{aggregated_price:.2f}")
        
        if current_price:
            levels = calculate_levels(pair, current_price, tp_percent, sl_percent)
            if levels:
                buy_signal = levels['rsi'] < RSI_OVERSOLD
                take_profit_signal = levels['rsi'] > RSI_OVERBOUGHT
                alert_cols = st.columns(3)
                rsi_color = "green" if levels['rsi'] < RSI_OVERSOLD else "red" if levels['rsi'] > RSI_OVERBOUGHT else "gray"
                alert_cols[0].markdown(f"<span style='color:{rsi_color};font-size:24px'>{levels['rsi']:.1f}</span>",
                                       unsafe_allow_html=True)
                alert_cols[0].caption("RSI (Oversold <30, Overbought >70)")
                alert_cols[1].metric("24h Range", f"£{levels['low']:,.2f}-£{levels['high']:,.2f}")
                alert_cols[2].metric("Volatility", f"£{levels['volatility']:,.2f}")
                
                with st.expander("Trading Strategy"):
                    action = ('🔥 Consider buying - Oversold market' if buy_signal else 
                              '💰 Consider profit taking - Overbought market' if take_profit_signal else 
                              '⏳ Hold - Neutral market conditions')
                    st.write(f"""
                    **Recommended Action:** {action}
                    
                    **Entry Zone:** £{levels['buy_zone']:,.2f}  
                    **Profit Target:** £{levels['take_profit']:,.2f} (+{tp_percent}%)  
                    **Stop Loss:** £{levels['stop_loss']:,.2f} (-{sl_percent}%)
                    """)
                    fig = go.Figure()
                    hist_data = get_realtime_data(pair)
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['Close'],
                        name='Price History',
                        line=dict(color='#1f77b4')
                    ))
                    fig.add_hline(y=levels['buy_zone'], line_dash="dot", annotation_text="Buy Zone", line_color="green")
                    fig.add_hline(y=levels['take_profit'], line_dash="dot", annotation_text="Profit Target", line_color="blue")
                    fig.add_hline(y=levels['stop_loss'], line_dash="dot", annotation_text="Stop Loss", line_color="red")
                    signals = pd.DataFrame(index=hist_data.index)
                    signals['Buy'] = hist_data['RSI'] < RSI_OVERSOLD
                    signals['Sell'] = hist_data['RSI'] > RSI_OVERBOUGHT
                    fig.add_trace(go.Scatter(
                        x=signals[signals['Buy']].index,
                        y=hist_data.loc[signals['Buy'], 'Close'],
                        mode='markers',
                        name='Buy Signals',
                        marker=dict(color='green', size=8, symbol='triangle-up')
                    ))
                    fig.add_trace(go.Scatter(
                        x=signals[signals['Sell']].index,
                        y=hist_data.loc[signals['Sell'], 'Close'],
                        mode='markers',
                        name='Sell Signals',
                        marker=dict(color='red', size=8, symbol='triangle-down')
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.write("## Position Builder")
                risk_amount = st.slider("Risk Amount (£)", 10.0, account_size, 100.0)
                position_size = risk_amount / abs(current_price - levels['stop_loss'])
                st.write(f"""
                **Suggested Position:**  
                - Size: {position_size:.6f} {pair.split('-')[0]}  
                - Value: £{(position_size * current_price):,.2f}  
                - Risk/Reward: 1:{risk_reward:.1f}
                """)
            else:
                st.error("Market data unavailable")
        else:
            st.warning("Waiting for price data...")
        
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
