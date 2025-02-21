import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime

# Configuration
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'BNB-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')

def get_realtime_price(pair):
    """Get real-time crypto prices in GBP without API key"""
    try:
        data = yf.Ticker(pair).history(period='1d', interval='1m')
        return data['Close'].iloc[-1] if not data.empty else None
    except:
        return None

def calculate_levels(pair):
    """Calculate trading levels using price action"""
    data = yf.download(pair, period='1d', interval='15m')
    if data.empty or len(data) < 20:
        return None
    
    current_price = data['Close'].iloc[-1].item()
    high = data['High'].iloc[-20:-1].max().item()
    low = data['Low'].iloc[-20:-1].min().item()
    
    return {
        'buy_zone': round((high + low) / 2, 2),
        'take_profit': round(high + (high - low) * 0.5, 2),
        'stop_loss': round(low - (high - low) * 0.25, 2),
        'current': current_price
    }

def calculate_position_size(account_size, risk_percent, stop_loss_distance):
    """Risk management calculator"""
    if stop_loss_distance <= 0:
        return 0
    risk_amount = account_size * (risk_percent / 100)
    return round(risk_amount / stop_loss_distance, 2)

def calculate_technical_indicators(pair):
    """Calculate additional technical indicators: SMA, RSI, and Bollinger Bands"""
    data = yf.download(pair, period='1d', interval='15m')
    if data.empty or len(data) < 20:
        return None
    
    # Short-term and long-term SMAs
    sma_short = data['Close'].rolling(window=10).mean().iloc[-1]
    sma_long = data['Close'].rolling(window=30).mean().iloc[-1]
    
    # Relative Strength Index (RSI) calculation (14 period)
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean().iloc[-1]
    avg_loss = loss.rolling(window=14).mean().iloc[-1]
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20 period SMA and standard deviation)
    sma20 = data['Close'].rolling(window=20).mean().iloc[-1]
    std20 = data['Close'].rolling(window=20).std().iloc[-1]
    upper_band = sma20 + (2 * std20)
    lower_band = sma20 - (2 * std20)
    
    return {
        'sma_short': round(sma_short, 2),
        'sma_long': round(sma_long, 2),
        'rsi': round(rsi, 2),
        'upper_band': round(upper_band, 2),
        'lower_band': round(lower_band, 2)
    }

def main():
    st.set_page_config(page_title="Free Crypto Trader", layout="centered")
    
    st.title("🇬🇧 Free Crypto Trading Bot")
    st.write("### Risk-Managed Trading Signals")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Balance (£):", 100, 1000000, 1000)
        risk_percent = st.slider("Risk Percentage:", 1, 10, 2)
    
    with col2:
        current_price = get_realtime_price(pair)
        if current_price:
            levels = calculate_levels(pair)
            
            if levels:
                position_size = calculate_position_size(
                    account_size,
                    risk_percent,
                    abs(current_price - levels['stop_loss'])
                )
                
                st.write("## Live Trading Signals")
                st.metric("Current Price", f"£{current_price:,.2f}")
                
                st.write(f"**Optimal Buy Zone:** £{levels['buy_zone']:,.2f}")
                st.write(f"**Take Profit Target:** £{levels['take_profit']:,.2f}")
                st.write(f"**Stop Loss Level:** £{levels['stop_loss']:,.2f}")
                st.write(f"**Recommended Position Size:** £{position_size:,.2f}")
                
                fig = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=current_price,
                    number={'prefix': "£", 'valueformat': ".2f"},
                    delta={'reference': levels['buy_zone'], 'relative': True},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("---")
                st.write("#### Risk Management Tips")
                st.write("1. Never risk more than 2% per trade")
                st.write("2. Always use stop losses")
                st.write("3. Verify levels across timeframes")
                st.write("4. Trade with the trend")
                
                # Display additional technical indicators
                indicators = calculate_technical_indicators(pair)
                if indicators:
                    st.write("#### Technical Indicators")
                    st.write(f"**Short-term SMA (10 periods):** £{indicators['sma_short']:,.2f}")
                    st.write(f"**Long-term SMA (30 periods):** £{indicators['sma_long']:,.2f}")
                    st.write(f"**RSI (14 periods):** {indicators['rsi']}")
                    st.write(f"**Bollinger Upper Band:** £{indicators['upper_band']:,.2f}")
                    st.write(f"**Bollinger Lower Band:** £{indicators['lower_band']:,.2f}")
            else:
                st.error("Insufficient market data for analysis")
        else:
            st.error("Couldn't fetch current prices. Try again later.")

if __name__ == "__main__":
    main()
