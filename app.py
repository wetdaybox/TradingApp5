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

# Configuration - Updated tickers with USD pairs and exchange rate
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
EXCHANGE_RATE_TICKER = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')

# Initialize session state
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'gbp_rate' not in st.session_state:
    st.session_state.gbp_rate = None

@st.cache_data(ttl=300)
def get_data(tickers, period='3d', interval='30m'):
    max_retries = 3
    for _ in range(max_retries):
        try:
            data = yf.download(tickers, period=period, interval=interval, group_by='ticker')
            if not data.empty and len(data) > 20:
                return data
            time.sleep(2)
        except Exception as e:
            st.error(f"Data fetch error: {str(e)}")
            time.sleep(5)
    return pd.DataFrame()

def convert_to_gbp(usd_prices, gbp_rate):
    return usd_prices * gbp_rate

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
        # Get data for both crypto and exchange rate
        tickers = [pair, EXCHANGE_RATE_TICKER]
        data = get_data(tickers)
        
        if data.empty or EXCHANGE_RATE_TICKER not in data:
            return None
            
        # Get GBP conversion rate (average of period)
        gbp_rate = data[EXCHANGE_RATE_TICKER]['Close'].mean()
        st.session_state.gbp_rate = gbp_rate  # Store for later use
        
        # Convert crypto data to GBP
        crypto_data = data[pair]
        crypto_data_gbp = crypto_data.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            crypto_data_gbp[col] = convert_to_gbp(crypto_data[col], gbp_rate)
        
        # Add technical indicators
        ta_features = ta.add_all_ta_features(
            crypto_data_gbp, open="Open", high="High", low="Low", 
            close="Close", volume="Volume", fillna=True
        )
        
        # Ensure required features exist
        required_features = ['momentum_rsi', 'trend_macd_diff', 'volatility_atr']
        if not all(f in ta_features.columns for f in required_features):
            return None

        # Multi-timeframe analysis
        hourly_data = get_data(tickers, period='5d', interval='1h')
        daily_data = get_data(tickers, period='30d', interval='1d')

        trends = {
            '30m': 'Bullish' if crypto_data_gbp['Close'].iloc[-50:].mean() > crypto_data_gbp['Close'].iloc[-100:-50].mean() else 'Bearish',
            '1h': 'Bullish' if convert_to_gbp(hourly_data[pair]['Close'], hourly_data[EXCHANGE_RATE_TICKER]['Close'].mean()).iloc[-50:].mean() > 
                   convert_to_gbp(hourly_data[pair]['Close'], hourly_data[EXCHANGE_RATE_TICKER]['Close'].mean()).iloc[-100:-50].mean() else 'Bearish',
            '1d': 'Bullish' if convert_to_gbp(daily_data[pair]['Close'], daily_data[EXCHANGE_RATE_TICKER]['Close'].mean()).iloc[-30:].mean() > 
                   convert_to_gbp(daily_data[pair]['Close'], daily_data[EXCHANGE_RATE_TICKER]['Close'].mean()).iloc[-60:-30].mean() else 'Bearish'
        }

        # ML Prediction
        if st.session_state.model is None:
            st.session_state.model = train_model(ta_features)
        
        prediction = 0
        if st.session_state.model:
            current_features = ta_features[required_features].iloc[-1].values.reshape(1, -1)
            prediction = st.session_state.model.predict(current_features)[0]

        return {
            'price': crypto_data_gbp['Close'].iloc[-1],
            'rsi': ta_features['momentum_rsi'].iloc[-1],
            'macd': ta_features['trend_macd_diff'].iloc[-1],
            'atr': ta_features['volatility_atr'].iloc[-1],
            'trends': trends,
            'prediction': 'Bullish' if prediction == 1 else 'Bearish',
            'levels': {
                'buy': crypto_data_gbp['Low'].iloc[-20:-1].min() * 0.98,
                'take_profit': [
                    crypto_data_gbp['High'].iloc[-20:-1].max() * 1.02,
                    crypto_data_gbp['High'].iloc[-20:-1].max() * 1.05
                ],
                'stop_loss': crypto_data_gbp['Low'].iloc[-20:-1].min() * 0.95
            }
        }
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

# Rest of the code remains the same with GBP formatting...

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
        
        if analysis and st.session_state.gbp_rate:
            # Display analysis with GBP formatting
            st.write("## Advanced Trading Signals")
            cols = st.columns(4)
            cols[0].metric("Current Price", f"£{analysis['price']:,.2f}")
            cols[1].metric("RSI (14)", f"{analysis['rsi']:.1f}", 
                          "Oversold" if analysis['rsi'] < 30 else "Overbought" if analysis['rsi'] > 70 else "Neutral")
            cols[2].metric("MACD", f"{analysis['macd']:.2f}", 
                          "Bullish" if analysis['macd'] > 0 else "Bearish")
            cols[3].metric("Volatility (ATR)", f"£{analysis['atr']:.2f}")
            
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
