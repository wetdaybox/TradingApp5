import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime
from typing import Dict, Optional

# Configuration with updated ticker symbols
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
UK_TIMEZONE = pytz.timezone('Europe/London')

# Configure Yahoo Finance with proper headers
session = yf.Ticker("", headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
})

@st.cache_data(ttl=300)
def download_data(pair: str, period: str = '1d', interval: str = '15m') -> pd.DataFrame:
    """Robust data download with error handling"""
    try:
        # Get current GBP/USD exchange rate
        fx_data = yf.download('GBPUSD=X', period='1d', interval='1m', session=session)
        fx_rate = 0.80  # Fallback rate
        if not fx_data.empty:
            fx_rate = fx_data['Close'].iloc[-1]
            
        data = yf.download(
            tickers=pair,
            period=period,
            interval=interval,
            session=session,
            progress=False
        )
        
        if not data.empty:
            # Convert USD prices to GBP using actual FX rate
            data[['Open', 'High', 'Low', 'Close']] *= fx_rate
            return data.round(2)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Data download error: {str(e)}")
        return pd.DataFrame()

def calculate_levels(pair: str) -> Optional[Dict[str, float]]:
    """Improved level calculation with validation"""
    data = download_data(pair)
    if data.empty or len(data) < 20:
        return None
    
    try:
        closed_data = data.iloc[:-1] if len(data) > 1 else data
        high = closed_data['High'].iloc[-20:].max()
        low = closed_data['Low'].iloc[-20:].min()
        current_price = data['Close'].iloc[-1]
        
        if any(np.isnan([high, low, current_price])):
            return None
            
        stop_loss = max(0.0, low - (high - low) * 0.25)
        
        return {
            'buy_zone': round((high + low) / 2, 2),
            'take_profit': round(high + (high - low) * 0.5, 2),
            'stop_loss': round(stop_loss, 2),
            'current': round(current_price, 2)
        }
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return None

# Rest of the functions remain the same...

def main():
    st.set_page_config(page_title="Crypto Trading Bot", layout="centered")
    st.title("🇬🇧 Free Crypto Trading Bot")
    st.write("### Risk-Managed Trading Signals")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        pair = selected_pair
        account_size = st.number_input("Account Balance (£)", 
                                     min_value=100, max_value=1000000, value=1000)
        risk_percent = st.slider("Risk Percentage", 1, 10, 2)
        base_currency = selected_pair.split('-')[0]

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

                st.write("## Trading Signals")
                st.metric("Current Price", f"£{current_price:,.2f}")
                st.caption(f"Last updated: {update_time}")
                
                cols = st.columns(3)
                cols[0].metric("Buy Zone", f"£{levels['buy_zone']:,.2f}")
                cols[1].metric("Take Profit", f"£{levels['take_profit']:,.2f}")
                cols[2].metric("Stop Loss", f"£{levels['stop_loss']:,.2f}")

                st.subheader("Position Sizing")
                st.write(f"**Recommended Size:** {position_size:,.4f} {base_currency}")
                st.write(f"**Position Value:** £{notional_value:,.2f}")
                st.progress(risk_percent/10, text=f"Risking {risk_percent}% of account")

                fig = go.Figure(go.Indicator(
                    mode="number+delta",
                    value=current_price,
                    number={'prefix': "£", 'valueformat': ".2f"},
                    delta={'reference': levels['buy_zone'], 'relative': False},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ))
                st.plotly_chart(fig, use_container_width=True)

                # Technical analysis section
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
