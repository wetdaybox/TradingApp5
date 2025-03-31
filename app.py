import warnings
import traceback
import logging
import time
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pytz
import requests
import os
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate critical dependencies
try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    assert joblib.__version__ >= '1.4.0'
    assert pd.__version__ >= '2.2.2'
except (ImportError, AssertionError) as e:
    st.error(f"Critical dependency error: {str(e)}")
    st.stop()

try:
    import feedparser
except ImportError:
    st.error("Missing 'feedparser'. Install with: pip install feedparser")
    feedparser = None

from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ======================================================
# Configuration with Validation
# ======================================================
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
REFRESH_INTERVAL = 60  # seconds
MODEL_PATH = "sgd_classifier.pkl"
INITIAL_MODEL_PARAMS = {'alpha': 0.0001, 'loss': 'log_loss', 'penalty': 'l2'}

# Technical thresholds validation
try:
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    assert RSI_OVERSOLD < RSI_OVERBOUGHT
except AssertionError:
    st.error("Invalid RSI configuration")
    st.stop()

# Session state initialization
session_defaults = {
    'atr_params': {'tp_multiplier': 3.0, 'sl_multiplier': 1.5},
    'classifier_params': INITIAL_MODEL_PARAMS,
    'manual_price': None,
    'last_update': datetime.now().strftime("%H:%M:%S"),
    'last_optimization_time': None,
    'error_count': 0
}

for key, val in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ======================================================
# Enhanced Technical Indicators
# ======================================================
def get_rsi(data, window=14):
    """Robust RSI calculation with error handling"""
    try:
        if len(data) < window + 1 or 'Close' not in data.columns:
            return pd.Series([np.nan]*len(data)), "Insufficient data for RSI"
            
        delta = data['Close'].diff().dropna()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window, min_periods=1).mean()
        avg_loss = loss.rolling(window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.clip(0, 100), None
    except Exception as e:
        logger.error(f"RSI calculation failed: {str(e)}")
        return pd.Series([np.nan]*len(data)), str(e)

def get_macd(data, fast=12, slow=26, signal=9):
    try:
        exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line, None
    except Exception as e:
        logger.error(f"MACD calculation failed: {str(e)}")
        return pd.Series(), pd.Series(), str(e)

def get_bollinger_bands(data, window=20, num_std=2):
    try:
        sma = data['Close'].rolling(window).mean()
        std = data['Close'].rolling(window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        return sma, upper, lower, None
    except Exception as e:
        logger.error(f"Bollinger Bands failed: {str(e)}")
        return pd.Series(), pd.Series(), pd.Series(), str(e)

# ======================================================
# Model Management System
# ======================================================
def initialize_model():
    """Create initial model if none exists"""
    try:
        base_model = SGDClassifier(**INITIAL_MODEL_PARAMS)
        joblib.dump(base_model, MODEL_PATH)
        logger.info("Initial model created")
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        st.error("Failed to initialize machine learning model")

def load_or_create_model():
    """Safe model loading with fallback"""
    if not os.path.exists(MODEL_PATH):
        st.warning("Initializing new trading model...")
        initialize_model()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return None

# ======================================================
# Enhanced Data Handling
# ======================================================
def get_fx_rate():
    """Get current GBP/USD exchange rate"""
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='5m', progress=False)
        if not fx_data.empty:
            return fx_data['Close'].iloc[-1]
        return 0.80  # Fallback rate
    except Exception as e:
        logger.error(f"FX rate error: {str(e)}")
        return 0.80

def safe_yfinance_fetch(pair, **kwargs):
    """Robust data fetching with retries"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = yf.download(pair, **kwargs)
            if not data.empty:
                return data
        except Exception as e:
            logger.warning(f"YFinance attempt {attempt+1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error("All YFinance attempts failed")
                return pd.DataFrame()
            time.sleep(2**attempt)

def get_realtime_data(pair):
    """Safe real-time data acquisition"""
    try:
        data = safe_yfinance_fetch(pair, period='7d', interval='5m', progress=False)
        if data.empty:
            st.error(f"No data returned for {pair}")
            return pd.DataFrame()
            
        # Calculate indicators
        data['RSI'], _ = get_rsi(data)
        macd_line, signal_line, _ = get_macd(data)
        data['MACD'] = macd_line
        data['MACD_Signal'] = signal_line
        _, upper_bb, lower_bb, _ = get_bollinger_bands(data)
        data['UpperBB'] = upper_bb
        data['LowerBB'] = lower_bb
        
        st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        return data
    except Exception as e:
        logger.error(f"Real-time data failed: {str(e)}")
        return pd.DataFrame()

def get_price_data(pair):
    """Get price data with manual override"""
    try:
        data = get_realtime_data(pair)
        fx_rate = get_fx_rate()
        
        if st.session_state.manual_price is not None:
            return st.session_state.manual_price, True
            
        if not data.empty and 'Close' in data.columns:
            primary_usd = data['Close'].iloc[-1]
            return primary_usd / fx_rate, False
            
        return None, False
    except Exception as e:
        logger.error(f"Price data failure: {str(e)}")
        return None, False

# ======================================================
# Trading Strategy Components
# ======================================================
def calculate_levels(pair, current_price, tp_multiplier, sl_multiplier, atr_lookback):
    """Calculate trading levels with validation"""
    try:
        data = get_realtime_data(pair)
        if data.empty:
            return None
            
        # Calculate ATR
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(atr_lookback).mean().iloc[-1]
        
        # Calculate levels
        return {
            'buy_zone': current_price - (atr * 0.5),
            'take_profit': current_price + (atr * tp_multiplier),
            'stop_loss': current_price - (atr * sl_multiplier),
            'rsi': data['RSI'].iloc[-1],
            'high': data['High'].iloc[-1],
            'low': data['Low'].iloc[-1],
            'volatility': (atr / current_price) * 100
        }
    except Exception as e:
        logger.error(f"Level calculation failed: {str(e)}")
        return None

# ======================================================
# Main Application
# ======================================================
def main():
    """Core application logic with safety checks"""
    try:
        st.title("🚀 Revolutionary Crypto Trading Bot")
        st.markdown("**Free-to-use, advanced crypto trading assistant**")
        
        # Model initialization
        model = load_or_create_model()
        if not model:
            return

        # Sidebar controls
        st.sidebar.header("Trading Parameters")
        pair = st.sidebar.selectbox("Select Asset:", CRYPTO_PAIRS)
        use_manual = st.sidebar.checkbox("Enter Price Manually")
        
        if use_manual:
            st.session_state.manual_price = st.sidebar.number_input(
                "Manual Price (£)", min_value=0.01, value=1000.0)
        else:
            st.session_state.manual_price = None

        # Price display
        current_price, is_manual = get_price_data(pair)
        if current_price:
            price_status = " (Manual)" if is_manual else ""
            st.metric(f"Current Price{price_status}", f"£{current_price:.4f}")

        # Main data display
        data = get_realtime_data(pair)
        if not data.empty:
            # Display price chart
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close']
            )])
            st.plotly_chart(fig, use_container_width=True)

            # Trading signals
            levels = calculate_levels(pair, current_price, 
                                     st.session_state.atr_params['tp_multiplier'],
                                     st.session_state.atr_params['sl_multiplier'], 14)
            if levels:
                st.subheader("Trading Signals")
                col1, col2, col3 = st.columns(3)
                col1.metric("Entry Zone", f"£{levels['buy_zone']:.4f}")
                col2.metric("Take Profit", f"£{levels['take_profit']:.4f}", delta="+{levels['take_profit']-current_price:.4f}")
                col3.metric("Stop Loss", f"£{levels['stop_loss']:.4f}", delta="-{current_price-levels['stop_loss']:.4f}")

        # Error handling display
        if st.session_state.error_count > 0:
            st.warning(f"Recovered from {st.session_state.error_count} errors")

    except Exception as e:
        st.session_state.error_count += 1
        logger.critical(f"Main failure: {str(e)}")
        st.error("Application Error - See details below")
        st.code(traceback.format_exc())
        st.button("Retry", on_click=lambda: st.experimental_rerun())

# ======================================================
# Application Bootstrap
# ======================================================
if __name__ == "__main__":
    st.set_page_config(
        page_title="Crypto Trading Bot",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st_autorefresh(interval=REFRESH_INTERVAL*1000, key="data_refresh")
    main()
