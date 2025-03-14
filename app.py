import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime
from typing import Dict, Optional

# Configuration
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'BNB-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')

@st.cache_data(ttl=300)
def get_realtime_price(pair: str) -> Optional[float]:
    """Get real-time crypto prices in GBP without API key"""
    try:
        data = yf.Ticker(pair).history(period='1d', interval='1m')
        return float(data['Close'].iloc[-1]) if not data.empty else None
    except Exception as e:
        st.error(f"Error fetching real-time price: {e}")
        return None

@st.cache_data(ttl=300)
def download_data(pair: str, period: str = '1d', interval: str = '15m') -> pd.DataFrame:
    """Download historical data using yfinance."""
    return yf.download(pair, period=period, interval=interval)

def calculate_levels(pair: str, current_price: float) -> Optional[Dict[str, float]]:
    """Calculate trading levels using price action"""
    data = download_data(pair, period='1d', interval='15m')
    if data.empty:
        return None
    
    # Exclude current forming candle
    closed_data = data.iloc[:-1] if len(data) > 1 else data
    if len(closed_data) < 20:
        return None
    
    high = closed_data['High'].iloc[-20:].max()
    low = closed_data['Low'].iloc[-20:].min()
    
    # Calculate stop loss, ensuring it's not negative
    stop_loss = low - (high - low) * 0.25
    stop_loss = max(0.0, stop_loss)  # Ensure stop loss is not negative
    
    return {
        'buy_zone': round((high + low) / 2, 2),
        'take_profit': round(high + (high - low) * 0.5, 2),
        'stop_loss': round(stop_loss, 2),
        'current': current_price
    }

def calculate_position_size(account_size: float, risk_percent: float, 
                           stop_loss_distance: float) -> float:
    """Risk management calculator: returns number of crypto units to buy"""
    if stop_loss_distance <= 0:
        return 0.0
    risk_amount = account_size * (risk_percent / 100)
    return round(risk_amount / stop_loss_distance, 4)

def calculate_technical_indicators(pair: str) -> Optional[Dict[str, float]]:
    """Calculate technical indicators: SMA, RSI, and Bollinger Bands"""
    data = download_data(pair, period='5d', interval='15m')
    if data.empty or len(data) < 30:
        return None
    
    # Short-term and long-term SMAs
    sma_short = data['Close'].rolling(window=10).mean().iloc[-1]
    sma_long = data['Close'].rolling(window=30).mean().iloc[-1]
    
    # RSI calculation using SMA
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean().iloc[-1]
    avg_loss = loss.rolling(window=14).mean().iloc[-1]
    
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma20 = data['Close'].rolling(window=20).mean().iloc[-1]
    std20 = data['Close'].rolling(window=20).std().iloc[-1]
    
    return {
        'sma_short': round(sma_short, 2),
        'sma_long': round(sma_long, 2),
        'rsi': round(rsi, 2),
        'upper_band': round(sma20 + 2 * std20, 2),
        'lower_band': round(sma20 - 2 * std20, 2)
    }

def main():
    st.set_page_config(page_title="Free Crypto Trader", layout="centered")
    st.title("🇬🇧 Free Crypto Trading Bot")
    st.write("### Risk-Managed Trading Signals")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Balance (£):", 
                                      min_value=100, max_value=1000000, value=1000)
        risk_percent = st.slider("Risk Percentage:", 1, 10, 2)
        base_currency = pair.split('-')[0]
    
    with col2:
        with st.spinner("Fetching market data..."):
            current_price = get_realtime_price(pair)
            if current_price:
                levels = calculate_levels(pair, current_price)
                update_time = datetime.now(UK_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
                
                if levels:
                    stop_loss_distance = abs(current_price - levels['stop_loss'])
                    position_size = calculate_position_size(
                        account_size, risk_percent, stop_loss_distance
                    )
                    notional_value = position_size * current_price

                    st.write("## Live Trading Signals")
                    st.metric("Current Price", f"£{current_price:,.2f}")
                    st.caption(f"Last updated: {update_time}")
                    
                    cols = st.columns(3)
                    cols[0].metric("Buy Zone", f"£{levels['buy_zone']:,.2f}")
                    cols[1].metric("Take Profit", f"£{levels['take_profit']:,.2f}")
                    cols[2].metric("Stop Loss", f"£{levels['stop_loss']:,.2f}")
                    
                    st.subheader("Position Sizing")
                    st.write(f"Recommended: **{position_size:,.4f} {base_currency}**")
                    st.write(f"Notional Value: £{notional_value:,.2f}")
                    
                    # Price indicator with delta
                    fig = go.Figure(go.Indicator(
                        mode="number+delta",
                        value=current_price,
                        number={'prefix': "£", 'valueformat': ".2f"},
                        delta={'reference': levels['buy_zone'], 'relative': True},
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Technical indicators
                    indicators = calculate_technical_indicators(pair)
                    if indicators:
                        st.subheader("Technical Analysis")
                        st.write(f"**10-period SMA:** £{indicators['sma_short']:,.2f}")
                        st.write(f"**30-period SMA:** £{indicators['sma_long']:,.2f}")
                        st.write(f"**RSI (14):** {indicators['rsi']}")
                        st.write(f"**Bollinger Upper:** £{indicators['upper_band']:,.2f}")
                        st.write(f"**Bollinger Lower:** £{indicators['lower_band']:,.2f}")
                        
                        # RSI interpretation
                        if indicators['rsi'] > 70:
                            st.warning("RSI indicates overbought conditions")
                        elif indicators['rsi'] < 30:
                            st.info("RSI indicates oversold conditions")
                else:
                    st.error("Insufficient market data for analysis")
            elif current_price is None:
                st.error("Failed to fetch market data. Please try again later.")

if __name__ == "__main__":
    main()
