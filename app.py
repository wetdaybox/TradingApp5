import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
from datetime import datetime

# 🇬🇧 Verified Configuration 🇬🇧
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'BNB-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')

# Core Session State
if 'last_update' not in st.session_state:
    st.session_state.last_update = "Loading..."
if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame()

def safe_data_fetch(pair):
    """Robust data fetching with timezone handling"""
    try:
        data = yf.download(pair, period='2d', interval='15m', progress=False)
        if not data.empty:
            # Proper timezone conversion
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC').tz_convert(UK_TIMEZONE)
            else:
                data.index = data.index.tz_convert(UK_TIMEZONE)
            return data[['Open', 'High', 'Low', 'Close']].copy()
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def main():
    st.set_page_config(page_title="🇬🇧 Crypto Trader Pro", layout="centered")
    st.title("🇬🇧 Personal Trading Assistant")
    
    # Asset Selection
    pair = st.selectbox("Select Asset:", CRYPTO_PAIRS)
    
    # Data Loading with Status
    with st.spinner("Fetching market data..."):
        if st.session_state.price_data.empty:
            st.session_state.price_data = safe_data_fetch(pair)
            st.session_state.last_update = datetime.now(UK_TIMEZONE).strftime("%H:%M:%S")
    
    # Display Core Information
    if not st.session_state.price_data.empty:
        st.caption(f"Last update: {st.session_state.last_update}")
        current_price = st.session_state.price_data['Close'].iloc[-1]
        
        # Price Display
        st.metric("Current Price", f"£{current_price:.2f}")
        
        # Price Chart
        try:
            fig = go.Figure(data=[
                go.Candlestick(
                    x=st.session_state.price_data.index,
                    open=st.session_state.price_data['Open'],
                    high=st.session_state.price_data['High'],
                    low=st.session_state.price_data['Low'],
                    close=st.session_state.price_data['Close'],
                    name='Price History'
                )
            ])
            fig.update_layout(
                xaxis_title='London Time',
                yaxis_title='Price (£)',
                height=400,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {str(e)}")
        
        # Trading Plan
        st.subheader("Trading Plan")
        latest = st.session_state.price_data.iloc[-1]
        st.write(f"""
        - **Entry Zone:** £{latest['Low'] * 0.98:.2f}
        - **Take Profit:** £{current_price * 1.15:.2f}
        - **Stop Loss:** £{current_price * 0.95:.2f}
        """)
    else:
        st.warning("Could not load market data. Please try again later.")

if __name__ == "__main__":
    main()
