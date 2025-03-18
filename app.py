import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
import numpy as np
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
# Helper: Price Formatter
# -------------------------------
def format_price(price):
    """Format price with dynamic precision"""
    if price < 10:
        return f"£{price:.8f}"
    elif price < 100:
        return f"£{price:.4f}"
    else:
        return f"£{price:.2f}"

# -------------------------------
# Original Indicator Calculation 
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
# Enhanced Data Fetching (Preserved Core Methodology)
# -------------------------------
@st.cache_data(ttl=25)  # Reduced from 30s
def get_realtime_data(pair):
    try:
        # Original download method preserved
        data = yf.download(pair, period='2d', interval='5m', progress=False)
        if not data.empty:
            data['RSI'] = get_rsi(data)
            st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        return data
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=15)  # Reduced from 60s + anomaly detection
def get_fx_rate():
    """Enhanced FX rate with spike detection"""
    try:
        # Preserve original download method
        fx_data = yf.download(FX_PAIR, period='1d', interval='5m', progress=False)
        if not fx_data.empty:
            # Analyze last 3 values for anomalies
            last_three = fx_data['Close'].iloc[-3:].values
            if len(last_three) >= 3:
                avg = last_three.mean()
                if abs((last_three[-1] - avg) / avg) > 0.005:
                    return np.median(last_three)
            return last_three[-1]
        return 0.80
    except Exception as e:
        st.error(f"FX error: {str(e)}")
        return 0.80

def cross_reference_price(pair):
    """Enhanced cross-check with interval alignment"""
    try:
        # Match main data's parameters
        ticker = yf.Ticker(pair)
        alt_data = ticker.history(period='2d', interval='5m')
        if not alt_data.empty:
            # Analyze last 3 closes for anomalies
            last_closes = alt_data['Close'].iloc[-3:].values
            if len(last_closes) >= 3:
                avg = last_closes.mean()
                if abs((last_closes[-1] - avg) / avg) > 0.015:
                    return np.median(last_closes)
            return last_closes[-1]
        return None
    except Exception as e:
        st.error(f"Alt data error: {str(e)}")
        return None

def get_price_data(pair):
    data = get_realtime_data(pair)
    fx_rate = get_fx_rate()
    if st.session_state.manual_price is not None:
        return st.session_state.manual_price, True
    if not data.empty and 'Close' in data.columns:
        return data['Close'].iloc[-1].item() / fx_rate, False
    return None, False

# -------------------------------
# Enhanced Levels Calculation
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
        st.error(f"Calculation error: {str(e)}")
        return None

# -------------------------------
# Main Application with Enhancements
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
        st.markdown(f"🕒 Last update: <span style='color:{recency_color}'>{st.session_state.last_update}</span>",
                    unsafe_allow_html=True)
        
        current_price, is_manual = get_price_data(pair)
        alt_price = cross_reference_price(pair)
        fx_rate = get_fx_rate()

        if current_price and alt_price and fx_rate:
            # Unified FX conversion
            main_price_gbp = current_price
            alt_price_gbp = alt_price / fx_rate
            
            # Dynamic weighting
            price_diff = abs(main_price_gbp - alt_price_gbp)
            price_diff_pct = (price_diff / main_price_gbp) * 100
            if price_diff_pct > 1:
                aggregated_price = (main_price_gbp * 0.7) + (alt_price_gbp * 0.3)
                st.metric("Price Discrepancy", f"{price_diff_pct:.2f}%", 
                          help="Significant difference between data sources")
            else:
                aggregated_price = (main_price_gbp + alt_price_gbp) / 2
            
            consistency_score = 100 - price_diff_pct
            st.metric("Data Consistency", 
                     f"{consistency_score:.1f}% Match",
                     help="Agreement between primary and secondary sources")
            
            st.write(f"**Aggregated Price:** {format_price(aggregated_price)}")
        
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
                alert_cols[1].metric("24h Range", f"{format_price(levels['low'])}-{format_price(levels['high'])}")
                alert_cols[2].metric("Volatility", f"{format_price(levels['volatility'])}")
                
                with st.expander("Trading Strategy"):
                    action = ('🔥 Consider buying - Oversold market' if buy_signal else 
                              '💰 Consider profit taking - Overbought market' if take_profit_signal else 
                              '⏳ Hold - Neutral market conditions')
                    st.write(f"""
                    **Recommended Action:** {action}
                    
                    **Entry Zone:** {format_price(levels['buy_zone'])}  
                    **Profit Target:** {format_price(levels['take_profit'])} (+{tp_percent}%)  
                    **Stop Loss:** {format_price(levels['stop_loss'])} (-{sl_percent}%)
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
                st.line_chart(bt_data['Portfolio'])
                st.write(f"**Total Return:** {total_return:.2f}%")
            else:
                st.error("Backtest could not be run due to insufficient data.")
    
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        **Data Consistency:**  
        New metric showing agreement between our primary data source and cross-checked values. Below 95% indicates significant discrepancy.
        
        **Price Discrepancy:**  
        Visible only when sources disagree by >1%. Helps identify potential data anomalies.
        
        **Aggregated Price:**  
        Now uses smart weighting - 70% main source when discrepancy exists, equal weighting otherwise.
        
        Other metrics retain original meaning with improved stability through:
        - FX rate spike detection
        - Price anomaly filtering
        - Synchronized data intervals
        """)
        
if __name__ == "__main__":
    main()
