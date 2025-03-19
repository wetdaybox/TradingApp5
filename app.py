import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pytz
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ======================================================
# Configuration
# ======================================================
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
REFRESH_INTERVAL = 60  # seconds between auto-refreshes

# ======================================================
# Session Initialization
# ======================================================
if 'manual_price' not in st.session_state:
    st.session_state.manual_price = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")

# ======================================================
# Helper Functions
# ======================================================
def format_price(price):
    """Format price with full precision based on magnitude."""
    if price < 10:
        return f"£{price:.8f}"
    elif price < 100:
        return f"£{price:.4f}"
    else:
        return f"£{price:.2f}"

# ======================================================
# Technical Indicator Functions
# ======================================================
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

def get_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def predict_next_return(data, lookback=20):
    """Simple ML signal: linear regression on log returns to predict next 5-min return (in %)."""
    if len(data) < lookback + 1:
        return 0
    data = data.copy()
    data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
    recent = data['LogReturn'].dropna().iloc[-lookback:]
    x = np.arange(len(recent))
    y = recent.values
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    return slope * 100

# ======================================================
# Data Fetching Functions (Proven Method)
# ======================================================
@st.cache_data(ttl=30)
def get_realtime_data(pair):
    """Fetch 2 days of 5-min data using yf.download. Ensure index is datetime."""
    try:
        data = yf.download(pair, period='2d', interval='5m', progress=False)
        if not data.empty:
            data.index = pd.to_datetime(data.index)
            data['RSI'] = get_rsi(data)
            st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        return data
    except Exception as e:
        st.error(f"Data error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_fx_rate():
    """Fetch FX rate using yf.download."""
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='5m', progress=False)
        return fx_data['Close'].iloc[-1].item() if not fx_data.empty else 0.80
    except Exception as e:
        st.error(f"FX error: {e}")
        return 0.80

def cross_reference_price(pair):
    """Fetch alternative price using yf.Ticker with period='1d', interval='1m'."""
    try:
        ticker = yf.Ticker(pair)
        alt_data = ticker.history(period='1d', interval='1m')
        if not alt_data.empty:
            return alt_data['Close'].iloc[-1].item()
        else:
            return None
    except Exception as e:
        st.error(f"Alternative data error: {e}")
        return None

def get_price_data(pair):
    data = get_realtime_data(pair)
    fx_rate = get_fx_rate()
    if st.session_state.manual_price is not None:
        return st.session_state.manual_price, True
    if not data.empty and 'Close' in data.columns:
        primary_usd = data['Close'].iloc[-1].item()
        return primary_usd / fx_rate, False
    return None, False

# ======================================================
# Calculate Levels and Backtesting
# ======================================================
def calculate_levels(pair, current_price, tp_percent, sl_percent):
    data = get_realtime_data(pair)
    if data.empty or len(data) < 288:
        return None
    try:
        full_day = data.iloc[-288:]
        recent_low = full_day['Low'].min().item()
        recent_high = full_day['High'].max().item()
        fx_rate = get_fx_rate()
        last_rsi = data['RSI'].iloc[-1]
        
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        vol = atr / fx_rate
        volatility = round(vol, 8) if vol < 1 else round(vol, 2)
        
        return {
            'buy_zone': round(recent_low * 0.98 / fx_rate, 2),
            'take_profit': round(current_price * (1 + tp_percent / 100), 2),
            'stop_loss': round(current_price * (1 - sl_percent / 100), 2),
            'rsi': round(last_rsi, 1),
            'high': round(recent_high / fx_rate, 2),
            'low': round(recent_low / fx_rate, 2),
            'volatility': volatility
        }
    except Exception as e:
        st.error(f"Calculation error: {e}")
        return None

def backtest_strategy(pair, tp_percent, sl_percent, initial_capital=1000):
    data = get_realtime_data(pair)
    if data.empty:
        return None
    fx_rate = get_fx_rate()
    df = data.copy()
    df['Price'] = df['Close'] / fx_rate
    df['Signal'] = 0
    df.loc[df['RSI'] < RSI_OVERSOLD, 'Signal'] = 1
    df.loc[df['RSI'] > RSI_OVERBOUGHT, 'Signal'] = -1
    position = 0
    cash = initial_capital
    portfolio = []
    for i in range(1, len(df)):
        if df['Signal'].iloc[i] == 1 and position == 0:
            position = cash / df['Price'].iloc[i]
            cash = 0
        elif df['Signal'].iloc[i] == -1 and position > 0:
            cash = position * df['Price'].iloc[i]
            position = 0
        portfolio.append(cash + position * df['Price'].iloc[i])
    df = df.iloc[1:].copy()
    df['Portfolio'] = portfolio
    total_return = (portfolio[-1] - initial_capital) / initial_capital * 100
    return df, total_return

# ======================================================
# Main Application
# ======================================================
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
        st.markdown(f"🕒 Last update: <span style='color:{recency_color}'>{st.session_state.last_update}</span>",
                    unsafe_allow_html=True)
        
        current_price, _ = get_price_data(pair)
        alt_price = cross_reference_price(pair)
        if current_price and alt_price:
            alt_price_gbp = alt_price / get_fx_rate()
            diff_pct = abs(current_price - alt_price_gbp) / current_price * 100
            st.metric("Price Diff (%)", f"{diff_pct:.2f}%")
            st.write(f"Alt Price (converted): {format_price(alt_price_gbp)}")
        
        # Display ML forecast signal
        data = get_realtime_data(pair)
        if not data.empty:
            ml_return = predict_next_return(data, lookback=20)
            ml_signal = "Buy" if ml_return > 0.05 else "Sell" if ml_return < -0.05 else "Hold"
            st.metric("ML Signal", ml_signal, delta=f"{ml_return:.2f}%")
        
        if current_price:
            levels = calculate_levels(pair, current_price, tp_percent, sl_percent)
            if levels:
                buy_signal = levels['rsi'] < RSI_OVERSOLD
                alert_cols = st.columns(3)
                rsi_color = "green" if levels['rsi'] < RSI_OVERSOLD else "red" if levels['rsi'] > RSI_OVERBOUGHT else "gray"
                alert_cols[0].markdown(f"<span style='color:{rsi_color};font-size:24px'>{levels['rsi']:.1f}</span>",
                                       unsafe_allow_html=True)
                alert_cols[0].caption("RSI (Oversold <30, Overbought >70)")
                alert_cols[1].metric("24h Range", f"{format_price(levels['low'])}-{format_price(levels['high'])}")
                alert_cols[2].metric("Volatility", f"{format_price(levels['volatility'])}")
                
                with st.expander("Trading Strategy"):
                    action = ('🔥 Consider buying - Oversold market' if buy_signal else 
                              '💰 Consider profit taking - Overbought market' if levels['rsi'] > RSI_OVERBOUGHT else 
                              '⏳ Hold - Neutral market conditions')
                    st.write(f"""
                    **Recommended Action:** {action}
                    
                    **Entry Zone:** {format_price(levels['buy_zone'])}  
                    **Profit Target:** {format_price(levels['take_profit'])} (+{tp_percent}%)  
                    **Stop Loss:** {format_price(levels['stop_loss'])} (-{sl_percent}%)
                    """)
                    
                    hist_data = get_realtime_data(pair)
                    if hist_data.empty:
                        st.error("Historical data not available for chart display.")
                    else:
                        # Reset index and find a datetime column robustly
                        hist_data_reset = hist_data.reset_index()
                        date_col = None
                        for col in hist_data_reset.columns:
                            if pd.api.types.is_datetime64_any_dtype(hist_data_reset[col]):
                                date_col = col
                                break
                        if date_col is None:
                            date_col = hist_data_reset.columns[0]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=hist_data_reset[date_col],
                            y=hist_data_reset['Close'],
                            name="Price History",
                            line=dict(color="#1f77b4")
                        ))
                        fig.add_hline(y=levels["buy_zone"], line_dash="dot",
                                      annotation_text="Buy Zone", line_color="green")
                        fig.add_hline(y=levels["take_profit"], line_dash="dot",
                                      annotation_text="Profit Target", line_color="blue")
                        fig.add_hline(y=levels["stop_loss"], line_dash="dot",
                                      annotation_text="Stop Loss", line_color="red")
                        fig.update_layout(
                            title=f"Historical Price Chart for {pair}",
                            xaxis_title="Date/Time",
                            yaxis_title="Price (£)",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                st.write("## Position Builder")
                risk_amount = st.slider("Risk Amount (£)", 10.0, account_size, 100.0)
                position_size = risk_amount / abs(current_price - levels["stop_loss"])
                st.write(f"""
                **Suggested Position:**  
                - Size: {position_size:.6f} {pair.split('-')[0]}  
                - Value: {format_price(position_size * current_price)}  
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
                st.line_chart(bt_data["Portfolio"])
                st.write(f"**Total Return:** {total_return:.2f}%")
            else:
                st.error("Backtest could not be run due to insufficient data.")
    
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        **Price Diff (%):**  
        The difference between the primary price (our proven method) and an alternative cross-referenced price. This helps verify the data’s accuracy.
        
        **ML Signal:**  
        A basic machine learning forecast (using a linear regression on log returns) predicts the next 5-minute return. Positive indicates Buy, negative indicates Sell, and near zero is Hold.
        
        **RSI (Relative Strength Index):**  
        A momentum indicator; values below 30 suggest an oversold asset, while above 70 indicate overbought.
        
        **24h Range:**  
        The lowest and highest prices in the past 24 hours.
        
        **Volatility:**  
        Derived from the Average True Range (ATR) over 14 periods, reflecting price fluctuations.
        
        **Trading Strategy:**  
        Uses RSI signals to generate entry, profit target, and stop loss levels.
        
        **Position Builder:**  
        Suggests a position size based on your risk amount and the gap between the current price and the stop loss.
        
        **Backtest Results:**  
        A historical simulation showing how a portfolio would have evolved using the RSI strategy.
        """)

if __name__ == "__main__":
    main()
