import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')

@st.cache_data(ttl=60)
def get_realtime_data(pair):
    """Get real-time market data with enhanced error handling"""
    try:
        data = yf.download(pair, period='1d', interval='1m', progress=False)
        return data[['Open', 'High', 'Low', 'Close']]
    except Exception as e:
        st.error(f"Market data error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_fx_rate():
    """Get GBP/USD FX rate with fallback"""
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='1m')
        return fx_data['Close'].iloc[-1] if not fx_data.empty else 0.80
    except:
        return 0.80  # Conservative fallback rate

def convert_to_gbp(usd_value, fx_rate):
    """Safe currency conversion"""
    return round(usd_value * fx_rate, 2)

def calculate_levels(data, fx_rate):
    """Enhanced level calculation with validation"""
    if data.empty or len(data) < 20:
        return None
    
    try:
        closed_data = data.iloc[:-1] if len(data) > 1 else data
        high_usd = closed_data['High'].iloc[-20:].max()
        low_usd = closed_data['Low'].iloc[-20:].min()
        current_usd = data['Close'].iloc[-1]

        stop_loss_usd = max(0.0, low_usd - (high_usd - low_usd) * 0.25)
        
        return {
            'buy_zone': convert_to_gbp((high_usd + low_usd)/2, fx_rate),
            'take_profit': convert_to_gbp(high_usd + (high_usd - low_usd)*0.5, fx_rate),
            'stop_loss': convert_to_gbp(stop_loss_usd, fx_rate),
            'current': convert_to_gbp(current_usd, fx_rate)
        }
    except Exception as e:
        st.error(f"Level calculation failed: {str(e)}")
        return None

def calculate_position_size(account_size, risk_percent, stop_loss_distance):
    """Robust position sizing"""
    try:
        risk_amount = account_size * (risk_percent / 100)
        return round(risk_amount / abs(stop_loss_distance), 4) if stop_loss_distance else 0
    except:
        return 0.0

def main():
    st.set_page_config(page_title="Pro Crypto Trader", layout="wide")
    
    st.title("🇬🇧 Professional Crypto Trading Terminal")
    st.write("### AI-Powered Risk Management System")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("CRYPTO PAIR", CRYPTO_PAIRS)
        account_size = st.number_input("ACCOUNT BALANCE (£)", 
                                     min_value=100, max_value=1000000, value=1000)
        risk_percent = st.slider("RISK PER TRADE (%)", 1, 10, 2)
        base_currency = pair.split('-')[0]
    
    with col2:
        with st.spinner("Analyzing market conditions..."):
            fx_rate = get_fx_rate()
            data = get_realtime_data(pair)
            
            if not data.empty:
                levels = calculate_levels(data, fx_rate)
                current_price = convert_to_gbp(data['Close'].iloc[-1], fx_rate)
                
                if levels:
                    stop_loss_distance = current_price - levels['stop_loss']
                    position_size = calculate_position_size(account_size, risk_percent, stop_loss_distance)
                    notional_value = position_size * current_price
                    
                    # Trading Signals Section
                    st.success("LIVE TRADING SIGNALS")
                    st.metric("CURRENT PRICE", f"£{current_price:,.2f}", 
                            delta=f"{stop_loss_distance:+.2f} from Stop Loss")
                    
                    # Key Levels Display
                    cols = st.columns(3)
                    cols[0].metric("BUY ZONE", f"£{levels['buy_zone']:,.2f}", 
                                 help="Optimal entry price range")
                    cols[1].metric("TAKE PROFIT", f"£{levels['take_profit']:,.2f}", 
                                 delta=f"+{levels['take_profit']-current_price:+.2f}")
                    cols[2].metric("STOP LOSS", f"£{levels['stop_loss']:,.2f}", 
                                  delta=f"-{current_price-levels['stop_loss']:+.2f}", 
                                  delta_color="inverse")
                    
                    # Position Management
                    st.subheader("POSITION MANAGEMENT")
                    st.write(f"""
                    - **Size:** {position_size:,.4f} {base_currency}
                    - **Value:** £{notional_value:,.2f}
                    - **Risk:** {risk_percent}% of account (£{account_size*risk_percent/100:,.2f})
                    """)
                    
                    # Price Indicator
                    fig = go.Figure(go.Indicator(
                        mode="number+delta",
                        value=current_price,
                        number={
                            'prefix': "£",
                            'valueformat': ",.2f",
                            'font': {'size': 40}
                        },
                        delta={
                            'reference': levels['buy_zone'],
                            'valueformat': ".2f",
                            'relative': False,
                            'increasing': {'color': "green"},
                            'decreasing': {'color': "red"}
                        },
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Insufficient data for analysis")
            else:
                st.error("Failed to fetch market data")

if __name__ == "__main__":
    main()
