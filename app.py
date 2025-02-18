import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime
from pytrends.request import TrendReq
from sklearn.ensemble import RandomForestClassifier
import ta
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
    max_retries = 3
    for _ in range(max_retries):
        try:
            data = yf.download(pair, period=period, interval=interval)
            if not data.empty and len(data) > 20:
                return data
            time.sleep(2)
        except Exception as e:
            st.error(f"Data fetch error: {str(e)}")
            time.sleep(5)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def train_model(data):
    try:
        features = data[['momentum_rsi', 'trend_macd_diff', 'volatility_atr']].dropna()
        target = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)[:-1]
        model = RandomForestClassifier(n_estimators=100)
        model.fit(features[:-1], target)
        return model
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None

def advanced_analysis(pair):
    try:
        data = get_data(pair)
        if data.empty:
            return None
            
        # Add technical indicators safely
        ta_features = ta.add_all_ta_features(
            data, open="Open", high="High", low="Low", 
            close="Close", volume="Volume", fillna=True
        )
        
        # Ensure required features exist
        required_features = ['momentum_rsi', 'trend_macd_diff', 'volatility_atr']
        if not all(f in ta_features.columns for f in required_features):
            return None

        # Multi-timeframe analysis
        hourly_data = get_data(pair, period='5d', interval='1h')
        daily_data = get_data(pair, period='30d', interval='1d')

        trends = {
            '30m': 'Bullish' if data['Close'].iloc[-50:].mean() > data['Close'].iloc[-100:-50].mean() else 'Bearish',
            '1h': 'Bullish' if hourly_data['Close'].iloc[-50:].mean() > hourly_data['Close'].iloc[-100:-50].mean() else 'Bearish',
            '1d': 'Bullish' if daily_data['Close'].iloc[-30:].mean() > daily_data['Close'].iloc[-60:-30].mean() else 'Bearish'
        }

        # ML Prediction
        if st.session_state.model is None:
            st.session_state.model = train_model(ta_features)
        
        if st.session_state.model:
            current_features = ta_features[required_features].iloc[-1].values.reshape(1, -1)
            prediction = st.session_state.model.predict(current_features)[0]
        else:
            prediction = 0

        return {
            'price': data['Close'].iloc[-1],
            'rsi': ta_features['momentum_rsi'].iloc[-1],
            'macd': ta_features['trend_macd_diff'].iloc[-1],
            'atr': ta_features['volatility_atr'].iloc[-1],
            'trends': trends,
            'prediction': 'Bullish' if prediction == 1 else 'Bearish',
            'levels': {
                'buy': data['Low'].iloc[-20:-1].min() * 0.98,
                'take_profit': [
                    data['High'].iloc[-20:-1].max() * 1.02,
                    data['High'].iloc[-20:-1].max() * 1.05
                ],
                'stop_loss': data['Low'].iloc[-20:-1].min() * 0.95
            }
        }
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

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

def main():
    st.set_page_config(page_title="Pro Crypto Trader", layout="wide")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Balance (£):", 100, 1000000, 1000)
        risk_percent = st.slider("Risk Percentage:", 1, 10, 2)
        
        sentiment = get_market_sentiment()
        if sentiment:
            st.write("### Market Sentiment")
            st.write(f"Bitcoin Interest: {sentiment['bitcoin']}/100")
            st.write(f"Ethereum Interest: {sentiment['ethereum']}/100")

    with col2:
        analysis = advanced_analysis(pair)
        
        if analysis:
            # Display analysis
            st.write("## Advanced Trading Signals")
            cols = st.columns(4)
            cols[0].metric("Current Price", f"£{analysis['price']:,.2f}")
            cols[1].metric("RSI (14)", f"{analysis['rsi']:.1f}", 
                          "Oversold" if analysis['rsi'] < 30 else "Overbought" if analysis['rsi'] > 70 else "Neutral")
            cols[2].metric("MACD", f"{analysis['macd']:.2f}", 
                          "Bullish" if analysis['macd'] > 0 else "Bearish")
            cols[3].metric("Volatility (ATR)", f"{analysis['atr']:.2f}")
            
            # Trading levels
            st.write("### Key Levels")
            level_cols = st.columns(3)
            level_cols[0].metric("Buy Zone", f"£{analysis['levels']['buy']:,.2f}")
            level_cols[1].metric("Take Profit", 
                               f"£{analysis['levels']['take_profit'][0]:,.2f} | £{analysis['levels']['take_profit'][1]:,.2f}")
            level_cols[2].metric("Stop Loss", f"£{analysis['levels']['stop_loss']:,.2f}")
            
            # Trend analysis
            st.write("### Multi-Timeframe Trends")
            trend_cols = st.columns(3)
            trend_cols[0].metric("30m Trend", analysis['trends']['30m'])
            trend_cols[1].metric("1h Trend", analysis['trends']['1h'])
            trend_cols[2].metric("Daily Trend", analysis['trends']['1d'])
            
            # ML Prediction
            st.write(f"#### AI Prediction: {analysis['prediction']}")

if __name__ == "__main__":
    main()
