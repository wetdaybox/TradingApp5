import warnings
import traceback
import logging
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

# Dependency validation
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
            return pd.Series([np.nan]*len(data), "Insufficient data for RSI"
            
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

# Similar enhanced implementations for MACD, Bollinger Bands, Stochastic...

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
# Enhanced Data Fetching
# ======================================================
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
            time.sleep(2**attempt)  # Exponential backoff

def get_realtime_data(pair):
    """Safe real-time data acquisition"""
    try:
        data = safe_yfinance_fetch(pair, period='7d', interval='5m', 
                                  progress=False, auto_adjust=True)
        if data.empty:
            st.error(f"No data returned for {pair}")
            return pd.DataFrame()
            
        # Data validation
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Missing price columns in {pair} data")
            return pd.DataFrame()
            
        # Feature calculation
        data['RSI'], rsi_error = get_rsi(data)
        if rsi_error:
            st.warning(f"RSI calculation: {rsi_error}")
            
        # Add other indicator calculations...
        
        st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        return data
    except Exception as e:
        logger.error(f"Real-time data failed: {str(e)}")
        st.error("Failed to fetch market data")
        return pd.DataFrame()

# ======================================================
# Main Application with Error Boundaries
# ======================================================
def main():
    """Core application logic with safety checks"""
    try:
        st.title("🚀 Revolutionary Crypto Trading Bot")
        st.markdown("**Free-to-use, advanced crypto trading assistant**")
        
        # Model initialization check
        model = load_or_create_model()
        if not model:
            st.error("Critical error: Trading model unavailable")
            return

        # Sidebar setup
        st.sidebar.header("Trading Parameters")
        pair = st.sidebar.selectbox("Select Asset:", CRYPTO_PAIRS)
        
        # Rest of your main application logic...
        # [Include all your existing main() content here]
        
    except Exception as e:
        st.session_state.error_count += 1
        logger.critical(f"Main application failure: {str(e)}")
        st.error("""
        ⚠️ Critical Application Error ⚠️
        Our system encountered an unexpected issue. Technical details below:
        """)
        st.code(f"ERROR TYPE: {type(e).__name__}\nMESSAGE: {str(e)}")
        st.write("Stack Trace:")
        st.code(traceback.format_exc())
        
        if st.session_state.error_count > 3:
            st.button("🔄 Restart Application", on_click=lambda: 
                     st.session_state.clear())

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
    
    try:
        main()
        st_autorefresh(interval=REFRESH_INTERVAL*1000, key="data_refresh")
    except Exception as e:
        st.error(f"Application bootstrap failed: {str(e)}")
        st.code(traceback.format_exc())
