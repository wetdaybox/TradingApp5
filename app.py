import streamlit as st
import pandas as pd
import yfinance as yf
import pytz
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

# Initialize logger first to catch startup errors
logger = logging.getLogger(__name__)
logger.addHandler(RotatingFileHandler('trading_bot.log', maxBytes=1e6, backupCount=3))
logger.setLevel(logging.INFO)
logger.info("------ Application Started ------")  # NEW: Startup confirmation

# Configuration (simplified for debugging)
CRYPTO_PAIRS = ['BTC-USD']  # NEW: Reduced to single pair for testing
FX_PAIR = 'USDGBP=X'
UK_TIMEZONE = pytz.timezone('Europe/London')

@st.cache_data(ttl=60)
def get_realtime_data(pair):
    """Safer data fetching with timezone debug"""
    try:
        logger.info(f"Fetching data for {pair}")  # NEW: Request tracking
        data = yf.download(pair, period='1d', interval='1m', progress=False)
        
        if data.empty:
            logger.warning("Received empty data frame")
            return pd.DataFrame()

        # Debug timezone conversion
        last_ts = data.index[-1].to_pydatetime().astimezone(UK_TIMEZONE)
        logger.info(f"Last data timestamp: {last_ts}")  # NEW: Time debug
        return data
        
    except Exception as e:
        logger.critical(f"Data fetch crashed: {str(e)}", exc_info=True)
        return pd.DataFrame()

def main():
    """Simplified main function for debugging"""
    st.set_page_config(page_title="Crypto Debug", layout="centered")
    st.title("Debug Version")
    
    try:
        logger.info("Rendering UI components")  # NEW: UI progress tracking
        st.write("## Basic Test")
        
        # Test data fetch
        test_data = get_realtime_data('BTC-USD')
        if not test_data.empty:
            st.write(f"Latest BTC Price: ${test_data['Close'].iloc[-1]:.2f}")
        else:
            st.warning("No data received")
            
    except Exception as e:
        logger.critical(f"UI rendering failed: {str(e)}", exc_info=True)
        st.error("Critical system failure - check logs")

if __name__ == "__main__":
    main()
