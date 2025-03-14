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
def download_data(pair: str, period: str = '1d', interval: str = '15m') -> pd.DataFrame:
    """Download historical data using yfinance (original method)"""
    try:
        return yf.download(pair, period=period, interval=interval)
    except Exception as e:
        st.error(f"Data download error: {str(e)}")
        return pd.DataFrame()

def calculate_levels(pair: str) -> Optional[Dict[str, float]]:
    """Original price level calculation with improvements"""
    data = download_data(pair, period='1d', interval='15m')
    if data.empty or len(data) < 20:
        return None
    
    # Original calculation method using last 20 completed candles
    closed_data = data.iloc[:-1] if len(data) > 1 else data
    high = closed_data['High'].iloc[-20:].max()
    low = closed_data['Low'].iloc[-20:].min()
    current_price = data['Close'].iloc[-1]
    
    # Improved validation
    if high <= low or current_price <= 0:
        return None
    
    stop_loss = max(0.0, low - (high - low) * 0.25)
    
    return {
        'buy_zone': round((high + low) / 2, 2),
        'take_profit': round(high + (high - low) * 0.5, 2),
        'stop_loss': round(stop_loss, 2),
        'current': round(current_price, 2)
    }

def calculate_position_size(account_size: float, risk_percent: float, 
                           stop_loss_distance: float) -> float:
    """Improved risk management calculator"""
    if stop_loss_distance <= 0 or account_size <= 0:
        return 0.0
    risk_amount = account_size * (risk_percent / 100)
    return round(risk_amount / stop_loss_distance, 4)

def calculate_technical_indicators(pair: str) -> Optional[Dict[str, float]]:
    """Technical analysis with original data source"""
    data = download_data(pair, period='5d', interval='15m')
    if data.empty or len(data) < 30:
        return None
    
    # SMA calculations
    sma_short = data['Close'].rolling(window=10).mean().iloc[-1]
    sma_long = data['Close'].rolling(window=30).mean().iloc[-1]
    
    # RSI calculation
    delta = data['Close'].diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    rs = avg_gain / avg_loss if avg_loss != 0 else 1
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma20 = data['Close'].rolling(20).mean().iloc[-1]
    std20 = data['Close'].rolling(20).std().iloc[-1]
    
    return {
        'sma_short': round(sma_short, 2),
        'sma_long': round(sma_long, 2),
        'rsi': round(rsi, 2),
        'upper_band': round(sma20 + 2*std20, 2),
        'lower_band': round(sma20 - 2*std20, 2)
    }

def main():
    st.set_page_config(page_title="Crypto Trading Bot", layout="centered")
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
        with st.spinner("Analyzing market data..."):
            levels = calculate_levels(pair)
            update_time = datetime.now(UK_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
            
            if levels:
                current_price = levels['current']
                stop_loss_distance = abs(current_price - levels['stop_loss'])
                position_size = calculate_position_size(
                    account_size, risk_percent, stop_loss_distance
                )
                notional_value = position_size * current_price

                # Display section
                st.write("## Trading Signals")
                st.metric("Current Price", f"£{current_price:,.2f}")
                st.caption(f"Last updated: {update_time}")
                
                # Key levels
                cols = st.columns(3)
                cols[0].metric("Buy Zone", f"£{levels['buy_zone']:,.2f}")
                cols[1].metric("Take Profit", f"£{levels['take_profit']:,.2f}")
                cols[2].metric("Stop Loss", f"£{levels['stop_loss']:,.2f}")

                # Position management
                st.subheader("Position Sizing")
                st.write(f"**Recommended Size:** {position_size:,.4f} {base_currency}")
                st.write(f"**Position Value:** £{notional_value:,.2f}")
                st.progress(risk_percent/10, text=f"Risking {risk_percent}% of account")

                # Price indicator
                fig = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=current_price,
                    number={'prefix': "£", 'valueformat': ".2f"},
                    delta={'reference': levels['buy_zone'], 'relative': False},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ))
                st.plotly_chart(fig, use_container_width=True)

                # Technical analysis
                indicators = calculate_technical_indicators(pair)
                if indicators:
                    st.subheader("Technical Indicators")
                    ta_cols = st.columns(2)
                    with ta_cols[0]:
                        st.write("**Moving Averages**")
                        st.metric("10-period SMA", f"£{indicators['sma_short']:,.2f}")
                        st.metric("30-period SMA", f"£{indicators['sma_long']:,.2f}")
                    with ta_cols[1]:
                        st.write("**Bollinger Bands**")
                        st.metric("Upper Band", f"£{indicators['upper_band']:,.2f}")
                        st.metric("Lower Band", f"£{indicators['lower_band']:,.2f}")
                    st.write(f"**RSI (14):** {indicators['rsi']:.1f} - ", 
                            "Overbought" if indicators['rsi'] > 70 else 
                            "Oversold" if indicators['rsi'] < 30 else "Neutral")
            else:
                st.error("Failed to calculate trading levels. Please try again later.")

if __name__ == "__main__":
    main()
