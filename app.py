import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'  # Represents USD per 1 GBP (e.g., 1.25 = $1.25 per £1)
UK_TIMEZONE = pytz.timezone('Europe/London')
FX_MIN = 1.20  # Minimum acceptable GBP/USD rate
FX_MAX = 1.40  # Maximum acceptable GBP/USD rate
DEFAULT_FX = 1.25  # Fallback rate
MAX_DATA_AGE_MIN = 5  # NEW: Reject data older than 5 minutes

@st.cache_data(ttl=60)
def get_realtime_data(pair):
    try:
        data = yf.download(pair, period='1d', interval='1m', progress=False)
        
        # NEW: Data freshness check
        if not data.empty:
            # Convert last timestamp to UTC
            last_ts = data.index[-1].to_pydatetime().replace(tzinfo=pytz.UTC)
            now = datetime.now(pytz.UTC)
            age_min = (now - last_ts).total_seconds() / 60
            
            if age_min > MAX_DATA_AGE_MIN:
                st.warning(f"Data is {age_min:.1f} minutes old - may be stale")
                return pd.DataFrame()
        
        return data
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_fx_rate():
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='1m')
        rate = fx_data['Close'].iloc[-1].item() if not fx_data.empty else DEFAULT_FX
        
        if not (FX_MIN <= rate <= FX_MAX):
            st.warning(f"Using default FX rate (invalid market rate: {rate:.2f})")
            rate = DEFAULT_FX
            
        return rate
    except Exception as e:
        st.warning(f"Using default FX rate (error: {str(e)})")
        return DEFAULT_FX

def get_current_price(pair):
    data = get_realtime_data(pair)
    fx_rate = get_fx_rate()
    if not data.empty:
        usd_price = data['Close'].iloc[-1].item()
        return round(usd_price / fx_rate, 2)
    return None

def calculate_levels(pair):
    data = get_realtime_data(pair)
    if data.empty or len(data) < 20:
        return None
    
    try:
        closed_data = data.iloc[:-1] if len(data) > 1 else data
        high = closed_data['High'].iloc[-20:].max().item()
        low = closed_data['Low'].iloc[-20:].min().item()
        fx_rate = get_fx_rate()
        
        buy_zone = (high + low) / 2 / fx_rate
        
        return {
            'buy_zone': round(buy_zone, 2),
            'take_profit': round(buy_zone * 1.05, 2),
            'stop_loss': round( (low - (high - low)*0.25) / fx_rate, 2),
            'current': round(data['Close'].iloc[-1].item() / fx_rate, 2)
        }
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return None

def calculate_position_size(account_size, risk_percent, stop_loss_distance):
    try:
        stop_loss_distance = float(stop_loss_distance)
        if stop_loss_distance <= 0:
            return 0.0
        risk_amount = account_size * (risk_percent / 100)
        return round(risk_amount / stop_loss_distance, 4)
    except Exception as e:
        st.error(f"Position error: {str(e)}")
        return 0.0

def main():
    st.set_page_config(page_title="Crypto Trader", layout="centered")
    st.title("🇬🇧 Free Crypto Trading Bot")
    st.write("### Risk-Managed Trading Signals")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Balance (£):", 
                                     min_value=100, max_value=1000000, value=1000)
        risk_percent = st.slider("Risk Percentage:", 1, 10, 2)
        
        if st.checkbox("🛠 Show FX debug info"):
            fx_rate = get_fx_rate()
            st.write(f"Current FX Rate (GBP/USD): {fx_rate:.4f}")
            st.write(f"Validation range: {FX_MIN}-{FX_MAX}")
    
    with col2:
        current_price = get_current_price(pair)
        if current_price:
            levels = calculate_levels(pair)
            if levels:
                try:
                    stop_loss_distance = abs(current_price - levels['stop_loss'])
                    position_size = calculate_position_size(account_size, risk_percent, stop_loss_distance)
                    notional_value = position_size * current_price
                    
                    st.write("## Live Trading Signals")
                    st.metric("Current Price", f"£{current_price:,.2f}")
                    
                    cols = st.columns(3)
                    cols[0].metric("Buy Zone", f"£{levels['buy_zone']:,.2f}")
                    cols[1].metric("Take Profit", f"£{levels['take_profit']:,.2f}")
                    cols[2].metric("Stop Loss", f"£{levels['stop_loss']:,.2f}")
                    
                    st.write(f"**Position Size:** {position_size:,.4f} {pair.split('-')[0]}")
                    st.write(f"**Position Value:** £{notional_value:,.2f}")

                    fig = go.Figure(go.Indicator(
                        mode="number+delta",
                        value=current_price,
                        number={'prefix': "£", 'valueformat': ".2f"},
                        delta={'reference': levels['buy_zone'], 'relative': False},
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Display error: {str(e)}")
            else:
                st.error("Insufficient market data for analysis")
        else:
            st.error("Couldn't fetch current prices. Try again later.")

if __name__ == "__main__":
    main()
