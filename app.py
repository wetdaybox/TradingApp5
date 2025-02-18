import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime
from pytrends.request import TrendReq
from sklearn.ensemble import RandomForestClassifier
from ta import add_all_ta_features
import time

# Configuration
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'BNB-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')

# Initialize session state
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_data(ttl=300)
def get_data(pair, period='3d', interval='30m'):
    data = yf.download(pair, period=period, interval=interval)
    return data.dropna()

@st.cache_data(ttl=3600)
def train_model(data):
    features = data[['momentum_rsi', 'trend_macd_diff', 'volatility_atr']].dropna()
    target = np.where(data['close'].shift(-1) > data['close'], 1, 0)[:-1]
    model = RandomForestClassifier(n_estimators=100)
    model.fit(features[:-1], target)
    return model

def advanced_analysis(pair):
    data = get_data(pair)
    
    # Add all technical indicators
    data = add_all_ta_features(data, 
                             open="Open", 
                             high="High", 
                             low="Low", 
                             close="Close", 
                             volume="Volume")
    
    # Multi-timeframe Analysis
    hourly_data = get_data(pair, period='5d', interval='1h')
    daily_data = get_data(pair, period='30d', interval='1d')
    
    # Trend Analysis
    trends = {
        '30m': 'Bullish' if data['Close'].iloc[-50:].mean() > data['Close'].iloc[-100:-50].mean() else 'Bearish',
        '1h': 'Bullish' if hourly_data['Close'].iloc[-50:].mean() > hourly_data['Close'].iloc[-100:-50].mean() else 'Bearish',
        '1d': 'Bullish' if daily_data['Close'].iloc[-30:].mean() > daily_data['Close'].iloc[-60:-30].mean() else 'Bearish'
    }
    
    # ML Prediction
    if st.session_state.model is None:
        st.session_state.model = train_model(data)
        
    current_features = data[['momentum_rsi', 'trend_macd_diff', 'volatility_atr']].iloc[-1].values.reshape(1, -1)
    prediction = st.session_state.model.predict(current_features)
    
    return {
        'price': data['Close'].iloc[-1],
        'rsi': data['momentum_rsi'].iloc[-1],
        'macd': data['trend_macd_diff'].iloc[-1],
        'atr': data['volatility_atr'].iloc[-1],
        'trends': trends,
        'prediction': 'Bullish' if prediction[0] == 1 else 'Bearish',
        'levels': {
            'buy': data['Low'].iloc[-20:-1].min() * 0.98,
            'take_profit': [
                data['High'].iloc[-20:-1].max() * 1.02,
                data['High'].iloc[-20:-1].max() * 1.05
            ],
            'stop_loss': data['Low'].iloc[-20:-1].min() * 0.95
        }
    }

@st.cache_data(ttl=3600)
def get_market_sentiment():
    try:
        pytrends = TrendReq(hl='en-GB', tz=0)
        pytrends.build_payload(['Bitcoin', 'Ethereum'], geo='GB', timeframe='now 7-d')
        trends = pytrends.interest_over_time()
        return {
            'bitcoin': trends['Bitcoin'].iloc[-1],
            'ethereum': trends['Ethereum'].iloc[-1]
        }
    except:
        return None

def risk_management():
    with st.sidebar:
        st.header("Risk Controls")
        return {
            'max_loss': st.slider("Max Daily Loss (%)", 1, 10, 2),
            'volatility': st.select_slider("Volatility Filter", options=['Low', 'Medium', 'High'], value='Medium'),
            'news_filter': st.checkbox("Enable News Filter", True)
        }

def update_portfolio(entry, exit, size):
    st.session_state.trades.append({
        'entry': entry,
        'exit': exit,
        'size': size,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def main():
    st.set_page_config(page_title="Pro Crypto Trader", layout="wide")
    
    # Risk Management Panel
    risk_params = risk_management()
    
    # Main Interface
    col1, col2 = st.columns([1, 3])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Balance (£):", 100, 1000000, 1000)
        risk_percent = st.slider("Risk Percentage:", 1, 10, 2)
        
        # Sentiment Analysis
        sentiment = get_market_sentiment()
        if sentiment:
            st.write("### Market Sentiment")
            st.write(f"Bitcoin Interest: {sentiment['bitcoin']}/100")
            st.write(f"Ethereum Interest: {sentiment['ethereum']}/100")

    with col2:
        analysis = advanced_analysis(pair)
        
        # Display Analysis
        st.write("## Advanced Trading Signals")
        cols = st.columns(4)
        cols[0].metric("Current Price", f"£{analysis['price']:,.2f}")
        cols[1].metric("RSI (14)", f"{analysis['rsi']:.1f}", 
                      "Oversold" if analysis['rsi'] < 30 else "Overbought" if analysis['rsi'] > 70 else "Neutral")
        cols[2].metric("MACD", f"{analysis['macd']:.2f}", 
                      "Bullish" if analysis['macd'] > 0 else "Bearish")
        cols[3].metric("Volatility (ATR)", f"{analysis['atr']:.2f}")
        
        # Trading Levels
        st.write("### Key Levels")
        level_cols = st.columns(3)
        level_cols[0].metric("Buy Zone", f"£{analysis['levels']['buy']:,.2f}")
        level_cols[1].metric("Take Profit", 
                           f"£{analysis['levels']['take_profit'][0]:,.2f} | £{analysis['levels']['take_profit'][1]:,.2f}")
        level_cols[2].metric("Stop Loss", f"£{analysis['levels']['stop_loss']:,.2f}")
        
        # Trend Analysis
        st.write("### Multi-Timeframe Trends")
        trend_cols = st.columns(3)
        trend_cols[0].metric("30m Trend", analysis['trends']['30m'])
        trend_cols[1].metric("1h Trend", analysis['trends']['1h'])
        trend_cols[2].metric("Daily Trend", analysis['trends']['1d'])
        
        # ML Prediction
        st.write(f"#### AI Prediction: {analysis['prediction']}")
        
        # Trading Panel
        st.write("### Execute Trade")
        size = account_size * (risk_percent / 100) / (analysis['price'] - analysis['levels']['stop_loss'])
        if st.button("Long Entry"):
            update_portfolio(analysis['price'], None, size)
        
        # Portfolio Display
        st.write("## Virtual Portfolio")
        if st.session_state.trades:
            portfolio = pd.DataFrame(st.session_state.trades)
            st.dataframe(portfolio.style.format({
                'entry': '£{:.2f}',
                'exit': '£{:.2f}',
                'size': '{:.4f}'
            }))
        else:
            st.write("No active trades")

if __name__ == "__main__":
    main()
