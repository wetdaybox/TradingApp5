import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
from datetime import datetime

# Configuration
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'BNB-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Initialize session state
if 'manual_price' not in st.session_state:
    st.session_state.manual_price = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now(UK_TIMEZONE).strftime("%H:%M:%S")

@st.cache_data(ttl=30)
def get_realtime_data(pair):
    """Get 48 hours of 5-minute data"""
    try:
        data = yf.download(pair, period='2d', interval='5m', progress=False, auto_adjust=True)
        if not data.empty:
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC').tz_convert(UK_TIMEZONE)
            else:
                data.index = data.index.tz_convert(UK_TIMEZONE)
            data['RSI'] = get_rsi(data)
            st.session_state.last_update = datetime.now(UK_TIMEZONE).strftime("%H:%M:%S")
        return data
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def get_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_price_data(pair):
    data = get_realtime_data(pair)
    
    if st.session_state.manual_price is not None:
        return float(st.session_state.manual_price), True
    
    if not data.empty:
        return data['Close'].iloc[-1].item(), False
    return None, False

def calculate_levels(pair, current_price):
    """24-hour range calculation"""
    data = get_realtime_data(pair)
    if data.empty or len(data) < 288:
        return None
    
    try:
        full_day_data = data.iloc[-288:]
        recent_low = full_day_data['Low'].min().item()
        recent_high = full_day_data['High'].max().item()
        last_rsi = data['RSI'].iloc[-1].item()
        
        return {
            'buy_zone': round(recent_low * 0.98, 2),
            'take_profit': round(current_price * 1.15, 2),
            'stop_loss': round(current_price * 0.95, 2),
            'rsi': round(last_rsi, 1),
            'high': round(recent_high, 2),
            'low': round(recent_low, 2)
        }
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Crypto Trader Pro", layout="centered")
    st.title("📈 Real-Time Crypto Assistant")
    
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
        
    with col2:
        st.caption(f"Last update: {st.session_state.last_update}")
        current_price, is_manual = get_price_data(pair)
        
        if current_price is not None:
            levels = calculate_levels(pair, current_price)
            if levels:
                buy_signal = levels['rsi'] < RSI_OVERSOLD
                take_profit_signal = levels['rsi'] > RSI_OVERBOUGHT
                
                alert_cols = st.columns(3)
                alert_cols[0].metric("RSI", f"{levels['rsi']}",
                                   delta="🔥 Buy Signal" if buy_signal else None)
                alert_cols[1].metric("24h Range", 
                                   f"£{levels['low']:,.2f}-£{levels['high']:,.2f}",
                                   help="24-hour trading range")
                alert_cols[2].metric("Next Target", f"£{levels['take_profit']:,.2f}",
                                   delta="💰 Take Profit" if take_profit_signal else None)
                
                with st.expander("Trading Strategy"):
                    st.write(f"""
                    **Recommended Action:**  
                    {'Consider buying - Oversold market' if buy_signal else 
                     'Consider profit taking - Overbought market' if take_profit_signal else 
                     'Hold - Neutral market conditions'}
                    
                    **Entry Zone:** £{levels['buy_zone']:,.2f}  
                    **Profit Target:** £{levels['take_profit']:,.2f} (+15%)  
                    **Stop Loss:** £{levels['stop_loss']:,.2f} (-5%)
                    """)
                    
                    data = get_realtime_data(pair)
                    if not data.empty:
                        fig = go.Figure(data=[
                            go.Scatter(
                                x=data.index,
                                y=data['Close'],
                                name='Price History'
                            ),
                            go.Scatter(
                                x=[datetime.now(UK_TIMEZONE)],
                                y=[current_price],
                                mode='markers',
                                marker=dict(color='red', size=10),
                                name='Current Price'
                            )
                        ])
                        fig.update_layout(
                            xaxis_title='London Time',
                            yaxis_title='Price (£)',
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                st.write("## Position Builder")
                risk_amount = st.slider("Risk Amount (£)", 10.0, account_size, 100.0)
                position_size = risk_amount / (current_price - levels['stop_loss'])
                st.write(f"""
                **Suggested Position:**  
                - Size: {position_size:.4f} {pair.split('-')[0]}  
                - Value: £{(position_size * current_price):,.2f}  
                - Risk/Reward: 1:3
                """)
                
            else:
                st.error("Market data unavailable")
        else:
            st.warning("Waiting for price data...")

if __name__ == "__main__":
    main()
