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

@st.cache_data(ttl=30)
def get_realtime_data(pair):
    """Get 48 hours of 5-minute data for accurate 24h range"""
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
        fx_data = yf.download(FX_PAIR, period='1d', interval='5m')
        return fx_data['Close'].iloc[-1].item() if not fx_data.empty else 0.80
    except Exception as e:
        st.error(f"FX error: {str(e)}")
        return 0.80

def get_rsi(data, window=14):
    if len(data) < window + 1:
        return pd.Series([None]*len(data))
    delta = data['Close'].diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss.replace(0, 0.0001)  # Prevent division by zero
    return 100 - (100 / (1 + rs))

def get_price_data(pair):
    data = get_realtime_data(pair)
    fx_rate = get_fx_rate()
    
    if st.session_state.manual_price is not None:
        return float(st.session_state.manual_price), True
    
    if not data.empty and 'Close' in data:
        return float(data['Close'].iloc[-1].item()) / fx_rate, False
    return None, False

def calculate_levels(pair, current_price, tp_percent, sl_percent):
    """Accurate 24-hour range calculation"""
    data = get_realtime_data(pair)
    if data.empty or len(data) < 288:  # 24h of 5m intervals (288 periods)
        return None
    
    try:
        full_day_data = data.iloc[-288:]  # Last 288 periods (24h)
        recent_low = float(full_day_data['Low'].min().item())
        recent_high = float(full_day_data['High'].max().item())
        fx_rate = get_fx_rate()
        last_rsi = float(data['RSI'].iloc[-1].item()) if 'RSI' in data else 50.0
        
        # Calculate volatility (ATR)
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = float(true_range.rolling(14).mean().iloc[-1].item())
        
        return {
            'buy_zone': round(recent_low * 0.98 / fx_rate, 4),
            'take_profit': round(current_price * (1 + tp_percent/100), 4),
            'stop_loss': round(current_price * (1 - sl_percent/100), 4),
            'rsi': round(last_rsi, 2),
            'high': round(recent_high / fx_rate, 4),
            'low': round(recent_low / fx_rate, 4),
            'volatility': round(atr / fx_rate, 4)
        }
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Crypto Trader Pro", layout="centered")
    st.title("📈 Real-Time Crypto Assistant")
    st_autorefresh(interval=REFRESH_INTERVAL*1000, key="main_refresh")
    
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
            
        account_size = st.number_input("Portfolio Value (£)", 
                                     min_value=100.0, value=1000.0, step=100.0)
        risk_profile = st.select_slider("Risk Profile:", 
                                      options=['Safety First', 'Balanced', 'High Risk'])
        risk_reward = st.slider("Risk/Reward Ratio", 1.0, 5.0, 3.0, 0.5)
        tp_percent = st.slider("Take Profit %", 1.0, 30.0, 15.0)
        sl_percent = st.slider("Stop Loss %", 1.0, 10.0, 5.0)
        
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
                alert_cols[0].markdown(f"<span style='color:{rsi_color};font-size:24px'>{levels['rsi']:.1f}</span>",
                                     unsafe_allow_html=True)
                alert_cols[0].caption("RSI (Oversold <30, Overbought >70)")
                
                alert_cols[1].metric("24h Range", 
                                   f"£{levels['low']:,.4f}-£{levels['high']:,.4f}")
                alert_cols[2].metric("Volatility", f"£{levels['volatility']:,.4f}")
                
                with st.expander("Trading Strategy"):
                    action = ('🔥 Consider buying - Oversold market' if buy_signal else 
                             '💰 Consider profit taking - Overbought market' if take_profit_signal else 
                             '⏳ Hold - Neutral market conditions')
                    
                    st.write(f"""
                    **Recommended Action:** {action}
                    
                    **Entry Zone:** £{levels['buy_zone']:,.4f}  
                    **Profit Target:** £{levels['take_profit']:,.4f} (+{tp_percent}%)  
                    **Stop Loss:** £{levels['stop_loss']:,.4f} (-{sl_percent}%)
                    """)
                    
                    fig = go.Figure()
                    hist_data = get_realtime_data(pair)
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['Close'],
                        name='Price History',
                        line=dict(color='#1f77b4')
                    ))
                    
                    # Strategy levels
                    fig.add_hline(y=levels['buy_zone'], line_dash="dot", 
                                annotation_text="Buy Zone", line_color="green")
                    fig.add_hline(y=levels['take_profit'], line_dash="dot",
                                annotation_text="Profit Target", line_color="blue")
                    fig.add_hline(y=levels['stop_loss'], line_dash="dot",
                                annotation_text="Stop Loss", line_color="red")
                    
                    # Historical signals
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
                position_size = risk_amount / ((current_price - levels['stop_loss']) * 
                                             (1 + levels['volatility']/100))
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

if __name__ == "__main__":
    main()
