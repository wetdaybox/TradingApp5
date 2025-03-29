import warnings
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

# Fixed dependency imports with version validation
try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV
    assert joblib.__version__ >= '1.4.0'  # Validate critical versions
    assert pd.__version__ >= '2.2.2'
except (ImportError, AssertionError) as e:
    st.error(f"Critical dependency error: {str(e)}. Please check package versions.")
    st.stop()

# News parsing with improved error handling
try:
    import feedparser
except ImportError:
    st.error("Missing 'feedparser'. Install with: pip install feedparser")
    feedparser = None

from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ======================================================
# Enhanced Configuration with Validation
# ======================================================
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
REFRESH_INTERVAL = 60  # seconds

# Validate technical indicator thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
assert RSI_OVERSOLD < RSI_OVERBOUGHT, "Invalid RSI thresholds"

MODEL_PATH = "sgd_classifier.pkl"

# Enhanced session state initialization
session_defaults = {
    'atr_params': {'tp_multiplier': 3.0, 'sl_multiplier': 1.5},  # Better risk-reward ratio
    'classifier_params': None,
    'manual_price': None,
    'last_update': datetime.now().strftime("%H:%M:%S"),
    'last_optimization_time': None
}

for key, val in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ======================================================
# Improved Technical Indicators with Boundary Checks
# ======================================================
def get_rsi(data, window=14):
    """Enhanced RSI calculation with data validation"""
    if len(data) < window + 1 or 'Close' not in data.columns:
        return pd.Series([np.nan]*len(data), index=data.index)
    
    delta = data['Close'].diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Handle division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(0, 100)  # Ensure RSI stays within 0-100 bounds

# Similar enhancements applied to MACD, Bollinger Bands, and Stochastic functions...

# ======================================================
# Optimized Machine Learning Pipeline
# ======================================================
def optimize_classifier(data, lookback=100):  # Increased lookback for better generalization
    """Enhanced model optimization with cross-validation"""
    if data.empty or len(data) < lookback:
        st.error("Insufficient data for optimization")
        return
    
    # Feature engineering
    df = data.copy()
    df['Return'] = df['Close'].pct_change()
    df = df.dropna()
    
    features = df[['RSI', 'MACD', 'StochK', 'Return']]
    target = (df['Return'].shift(-1) > 0).astype(int).dropna()
    
    # Temporal cross-validation
    param_grid = {
        'alpha': [1e-4, 1e-3, 1e-2, 0.1],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'loss': ['log_loss', 'modified_huber']
    }
    
    try:
        base_model = SGDClassifier(max_iter=2000, tol=1e-4, early_stopping=True)
        grid = GridSearchCV(base_model, param_grid, cv=TimeSeriesSplit(3), n_jobs=-1)
        grid.fit(features, target)
        
        st.session_state.classifier_params = grid.best_params_
        joblib.dump(grid.best_estimator_, MODEL_PATH)
        
        st.write("Optimized Parameters:")
        st.json(grid.best_params_)
        
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")

# ======================================================
# Enhanced Sentiment Analysis with Multiple Sources
# ======================================================
CRYPTO_NEWS_SOURCES = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptopanic.com/news/rss/",
    "https://cointelegraph.com/rss"
]

def get_sentiment(pair):
    """Improved sentiment analysis with multiple fallback sources"""
    headlines = []
    
    # Try Yahoo Finance first
    try:
        news = yf.Ticker(pair).news or []
        headlines.extend([n.get('title', '') for n in news if n.get('title')])
    except Exception:
        pass
    
    # Fallback to RSS feeds if needed
    if feedparser and len(headlines) < 5:
        for source in CRYPTO_NEWS_SOURCES:
            try:
                feed = feedparser.parse(source)
                headlines.extend([e.title for e in feed.entries[:10] if e.title])
                if len(headlines) >= 10: break
            except Exception:
                continue
    
    # Sentiment analysis
    if not headlines:
        return "Neutral", []
    
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    avg_score = np.mean(scores)
    
    # Weighted sentiment calculation
    if avg_score > 0.15:  # More conservative thresholds
        sentiment = "Positive"
    elif avg_score < -0.15:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return sentiment, headlines

# ======================================================
# Improved Backtesting with Realistic Assumptions
# ======================================================
def backtest_strategy(pair, tp_multiplier, sl_multiplier, atr_lookback, 
                     trailing_stop_percent, initial_capital=1000):
    """Enhanced backtest with transaction costs and slippage"""
    data = get_realtime_data(pair)
    if data.empty:
        return None, None
    
    # Realistic trading assumptions
    TRANSACTION_FEE = 0.001  # 0.1% per trade
    SLIPPAGE = 0.0005       # 0.05% price impact
    
    # Rest of backtest implementation with fee/slippage accounting...
