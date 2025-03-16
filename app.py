import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
from datetime import datetime, timedelta

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
REFRESH_MINUTES = 5  # Manual refresh interval

# Initialize session state
if 'manual_price' not in st.session_state:
    st.session_state.manual_price = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")

@st.cache_data(ttl=300)
def get_realtime_data(pair):
    """Get 48 hours of 5-minute data with timezone conversion"""
    try:
        data = yf.download(pair, period='2d', interval='5m', progress=False)
        if not data.empty:
            data.index = data.index.tz_localize('UTC').tz_convert(UK_TIMEZONE)
            data['RSI'] = get_rsi(data)
            st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        return data
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_fx_history(start, end):
    """Get historical FX rates for specific timeframe"""
    try:
        return yf.download(FX_PAIR, start=start, end=end, interval='5m')
    except Exception as e:
        st.error(f"FX history error: {str(e)}")
        return pd.DataFrame()

def get_rsi(data, window=14):
    """Accurate RSI calculation"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_price_data(pair):
    """Get current price with validation"""
    data = get_realtime_data(pair)
    if data.empty:
        return None, False
    
    try:
        current_fx = yf.download(FX_PAIR, period='1d', interval='1m')['Close'].iloc[-1]
        return data['Close'].iloc[-1] / current_fx, False
    except:
        return None, False

def calculate_levels(pair, current_price):
    """Precise 24h range calculation with historical FX conversion"""
    data = get_realtime_data(pair)
    if data.empty:
        return None
    
    try:
        # Get exact 24h window
        end_time = data.index[-1]
        start_time = end_time - timedelta(hours=24)
        mask = (data.index >= start_time) & (data.index <= end_time)
        crypto_data = data.loc[mask].copy()
        
        # Get matching FX data
        fx_data = get_fx_history(start_time, end_time)
        if fx_data.empty:
            return None
            
        # Merge datasets
        merged_data = crypto_data.merge(fx_data['Close'], 
                                      left_index=True, right_index=True,
                                      suffixes=('_crypto', '_fx'))
        
        # Convert prices using historical FX rates
        merged_data['Low_GBP'] = merged_data['Low'] / merged_data['Close_fx']
        merged_data['High_GBP'] = merged_data['High'] / merged_data['Close_fx']
        
        recent_low = merged_data['Low_GBP'].min()
        recent_high = merged_data['High_GBP'].max()
        current_rsi = merged_data['RSI'].iloc[-1] if not merged_data.empty else 50

        return {
            'buy_zone': round(recent_low * 0.98, 2),
            'take_profit': round(current_price * 1.15, 2),
            'stop_loss': round(current_price * 0.95, 2),
            'rsi': round(current_rsi, 1),
            'high': round(recent_high, 2),
            'low': round(recent_low, 2),
            'merged_data': merged_data
        }
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Crypto Trader Pro", layout="centered")
    st.title("📈 Real-Time Crypto Assistant")
    
    # Manual refresh system
    if st.button('🔄 Refresh Data'):
        st.cache_data.clear()
    
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
        st.caption(f"Last update: {st.session_state.last_update} ({UK_TIMEZONE.zone})")
        current_price, is_manual = get_price_data(pair)
        
        if current_price:
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
                    
                    if not levels['merged_data'].empty:
                        fig = go.Figure(data=[
                            go.Scatter(
                                x=levels['merged_data'].index,
                                y=levels['merged_data']['Close_crypto'] / levels['merged_data']['Close_fx'],
                                name='Price History',
                                line=dict(color='#1f77b4')
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
                            hovermode="x unified",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Historical data unavailable for chart")
                
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
