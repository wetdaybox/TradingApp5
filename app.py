import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, timedelta
from pytrends.request import TrendReq
from sklearn.ensemble import RandomForestClassifier
import ta
import time
import requests

# Configuration with free data sources
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FALLBACK_RATE = 0.79  # Hardcoded GBP/USD rate
UK_TIMEZONE = pytz.timezone('Europe/London')

# Initialize session state with fallback data
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'gbp_rate' not in st.session_state:
    st.session_state.gbp_rate = FALLBACK_RATE
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = {}

def safe_yfinance_fetch(ticker, period='3d', interval='30m'):
    """Fetch data with aggressive error handling"""
    max_retries = 3
    backoff_times = [2, 4, 8]
    
    for attempt in range(max_retries):
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                timeout=10
            )
            if not data.empty:
                return data
        except Exception as e:
            st.warning(f"Attempt {attempt+1} failed for {ticker}")
            time.sleep(backoff_times[attempt])
    
    st.error(f"Failed to fetch {ticker} after {max_retries} attempts")
    return pd.DataFrame()

def get_exchange_rate():
    """Get GBP/USD rate with multiple fallback strategies"""
    try:
        rate_data = safe_yfinance_fetch('GBPUSD=X', period='1d')
        if not rate_data.empty:
            return rate_data['Close'].iloc[-1]
    except:
        pass
    
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=5)
        return response.json()['rates']['GBP']
    except:
        return FALLBACK_RATE

def convert_to_gbp(usd_prices):
    """Safe conversion with rate validation"""
    rate = get_exchange_rate()
    if 0.5 < rate < 2.0:
        return usd_prices * rate
    return usd_prices * FALLBACK_RATE

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
        # Try to get fresh data
        crypto_data = safe_yfinance_fetch(pair)
        if crypto_data.empty:
            st.warning("Using cached data")
            crypto_data = st.session_state.cached_data.get(pair, pd.DataFrame())
            if crypto_data.empty:
                return None
        
        # Store successful data
        st.session_state.cached_data[pair] = crypto_data
        
        # Convert to GBP
        crypto_gbp = crypto_data.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            crypto_gbp[col] = convert_to_gbp(crypto_data[col])
        
        # Technical analysis
        ta_features = ta.add_all_ta_features(
            crypto_gbp, open="Open", high="High", low="Low", 
            close="Close", volume="Volume", fillna=True
        )

        required_features = ['momentum_rsi', 'trend_macd_diff', 'volatility_atr']
        if not all(f in ta_features.columns for f in required_features):
            return None

        # Prediction model
        if st.session_state.model is None:
            st.session_state.model = train_model(ta_features)
        
        prediction = 0
        if st.session_state.model:
            current_features = ta_features[required_features].iloc[-1].values.reshape(1, -1)
            try:
                prediction = st.session_state.model.predict(current_features)[0]
            except:
                prediction = 0

        return {
            'price': crypto_gbp['Close'].iloc[-1],
            'rsi': ta_features['momentum_rsi'].iloc[-1],
            'macd': ta_features['trend_macd_diff'].iloc[-1],
            'atr': ta_features['volatility_atr'].iloc[-1],
            'prediction': 'Bullish' if prediction == 1 else 'Bearish',
            'levels': {
                'buy': crypto_gbp['Low'].iloc[-20:-1].min() * 0.98,
                'take_profit': crypto_gbp['High'].iloc[-20:-1].max() * 1.02,
                'stop_loss': crypto_gbp['Low'].iloc[-20:-1].min() * 0.95
            }
        }
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Free Crypto Trader", layout="wide")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        st.write(f"Current GBP Rate: £1 = ${st.session_state.gbp_rate:.2f}")
        if st.button("Refresh Data"):
            st.session_state.cached_data.clear()
            st.rerun()
    
    with col2:
        analysis = advanced_analysis(pair)
        
        if analysis:
            st.write("## Trading Signals")
            cols = st.columns(3)
            cols[0].metric("Price", f"£{analysis['price']:,.2f}")
            cols[1].metric("RSI", f"{analysis['rsi']:.1f}", 
                          "Oversold" if analysis['rsi'] <30 else "Overbought" if analysis['rsi']>70 else "Neutral")
            cols[2].metric("Trend", analysis['prediction'])
            
            st.write("### Trading Levels")
            level_cols = st.columns(3)
            level_cols[0].metric("Buy Zone", f"£{analysis['levels']['buy']:,.2f}")
            level_cols[1].metric("Take Profit", f"£{analysis['levels']['take_profit']:,.2f}")
            level_cols[2].metric("Stop Loss", f"£{analysis['levels']['stop_loss']:,.2f}")
        else:
            st.warning("Data temporarily unavailable. Try refreshing.")

if __name__ == "__main__":
    main()
