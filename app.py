import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime
import logging  # NEW
from logging.handlers import RotatingFileHandler  # NEW

# NEW: Configure logging
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('trading_bot.log', maxBytes=1e6, backupCount=3)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
DEFAULT_FX = 0.80  # NEW
FX_MIN, FX_MAX = 0.70, 1.00  # NEW
MAX_POSITION_SIZE = 1000  # NEW

@st.cache_data(ttl=60)
def get_realtime_data(pair):
    """Get real-time crypto prices"""
    try:
        data = yf.download(pair, period='1d', interval='1m', progress=False)
        
        # NEW: Data freshness check
        if not data.empty:
            last_ts = data.index[-1].to_pydatetime().astimezone(UK_TIMEZONE)
            now = datetime.now(UK_TIMEZONE)
            if (now - last_ts).total_seconds() > 300:
                logger.warning(f"Stale data for {pair}: {last_ts} vs {now}")
                return pd.DataFrame()
        return data
    except Exception as e:
        logger.error(f"Data fetch failed for {pair}", exc_info=True)  # NEW
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_fx_rate():
    """Get current GBP/USD exchange rate"""
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='1m')
        if fx_data.empty:
            return DEFAULT_FX
            
        rate = fx_data['Close'].iloc[-1].item()
        # NEW: FX validation
        return rate if FX_MIN <= rate <= FX_MAX else DEFAULT_FX
    except Exception as e:
        logger.warning(f"Using default FX rate: {str(e)}")  # NEW
        return DEFAULT_FX

def get_current_price(pair):
    """Get converted GBP price"""
    data = get_realtime_data(pair)
    fx_rate = get_fx_rate()
    
    if not data.empty:
        usd_price = data['Close'].iloc[-1].item()
        return round(usd_price / fx_rate, 2)
    return None

def calculate_levels(pair):
    """Calculate trading levels"""
    data = get_realtime_data(pair)
    if data.empty or len(data) < 20:
        return None
    
    try:
        fx_rate = get_fx_rate()  # NEW: Cache FX rate
        closed_data = data.iloc[:-1] if len(data) > 1 else data
        high = closed_data['High'].iloc[-20:].max().item()
        low = closed_data['Low'].iloc[-20:].min().item()
        current_price = data['Close'].iloc[-1].item() / fx_rate
        
        stop_loss = max(0.0, low - (high - low) * 0.25)
        
        return {
            'buy_zone': round((high + low) / 2 / fx_rate, 2),
            'take_profit': round(high + (high - low) * 0.5 / fx_rate, 2),
            'stop_loss': round(stop_loss / fx_rate, 2),
            'current': round(current_price, 2)
        }
    except Exception as e:
        logger.error(f"Level calc failed for {pair}", exc_info=True)  # NEW
        st.error(f"Calculation error: {str(e)}")
        return None

def calculate_position_size(account_size, risk_percent, stop_loss_distance):
    """Risk management calculator"""
    try:
        stop_loss_distance = float(stop_loss_distance)
        if stop_loss_distance <= 0:
            return 0.0
        risk_amount = account_size * (risk_percent / 100)
        position = round(risk_amount / stop_loss_distance, 4)
        return min(position, MAX_POSITION_SIZE)  # NEW: Position cap
    except Exception as e:
        logger.error(f"Position calc error: {str(e)}")  # NEW
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
    
    with col2:
        current_price = get_current_price(pair)
        status = st.empty()  # NEW: Status container
        if current_price:
            levels = calculate_levels(pair)
            if levels:
                try:
                    stop_loss_distance = abs(current_price - levels['stop_loss'])
                    position_size = calculate_position_size(
                        account_size, risk_percent, stop_loss_distance
                    )
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
                    
                    # NEW: System status
                    status.success("✅ System operational - data current")
                except Exception as e:
                    logger.error(f"Display error: {str(e)}", exc_info=True)  # NEW
                    status.error("❌ System error - check logs")
                    st.error(f"Display error: {str(e)}")
            else:
                status.warning("⚠️ Partial data - refresh required")  # NEW
                st.error("Insufficient market data for analysis")
        else:
            status.warning("⚠️ Partial data - refresh required")  # NEW
            st.error("Couldn't fetch current prices. Try again later.")

if __name__ == "__main__":
    main()
