import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
from datetime import datetime

# 🇬🇧 Configuration 🇬🇧
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'BNB-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now(UK_TIMEZONE).strftime("%H:%M:%S")

@st.cache_data(ttl=30)
def get_realtime_data(pair):
    try:
        data = yf.download(pair, period='2d', interval='5m', progress=False)
        if not data.empty:
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
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    return 100 - (100 / (1 + (avg_gain / avg_loss)))

def main():
    st.set_page_config(page_title="🇬🇧 Crypto Trader Pro", layout="centered")
    st.title("🇬🇧 Personal Trading Assistant")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Asset:", CRYPTO_PAIRS)
        st.caption(f"Last update: {st.session_state.last_update}")
        
    with col2:
        data = get_realtime_data(pair)
        
        if not data.empty:
            current_price = data['Close'].iloc[-1].item()
            levels = {
                'current': current_price,
                'rsi': data['RSI'].iloc[-1].item(),
                'high': data['High'].max().item(),
                'low': data['Low'].min().item()
            }
            
            # Trading Signals
            st.subheader("Trading Signals")
            st.metric("RSI", f"{levels['rsi']:.1f}", 
                     delta="Oversold" if levels['rsi'] < RSI_OVERSOLD else 
                     "Overbought" if levels['rsi'] > RSI_OVERBOUGHT else "Neutral")
            
            # Price Chart
            valid_data = data[['Close']].dropna()
            if not valid_data.empty:
                fig = go.Figure(data=[
                    go.Scatter(
                        x=valid_data.index.astype(str),
                        y=valid_data['Close'],
                        name='Price History',
                        line=dict(color='#1f77b4', width=2)
                ])
                fig.update_layout(
                    xaxis_title='London Time',
                    yaxis_title='Price (£)',
                    hovermode="x unified",
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid price data for chart")
            
            # Trading Plan
            st.subheader("Trading Plan")
            st.write(f"""
            **Entry Zone:** £{levels['low'] * 0.98:.2f}  
            **Take Profit:** £{current_price * 1.15:.2f}  
            **Stop Loss:** £{current_price * 0.95:.2f}
            """)
            
        else:
            st.warning("Loading market data...")

if __name__ == "__main__":
    main()
