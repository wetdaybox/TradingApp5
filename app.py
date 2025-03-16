import streamlit as st
import pandas as pd
import yfinance as yf
import pytz
from datetime import datetime
import logging

# Configure logging to show errors in Streamlit Cloud
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Critical change for cloud logging
logger.setLevel(logging.INFO)

# Simplified configuration
CRYPTO_PAIRS = ['BTC-USD']  # Single pair for testing
FX_PAIR = 'USDGBP=X'
UK_TIMEZONE = pytz.timezone('Europe/London')

@st.cache_data(ttl=60, show_spinner=False)
def get_realtime_data(pair):
    """Safe data fetching with cloud-friendly time handling"""
    try:
        logger.info(f"Fetching data for {pair}")
        data = yf.download(pair, period='1d', interval='1m', progress=False)
        
        if data.empty:
            logger.warning("Empty data received")
            return pd.DataFrame()

        # Cloud-friendly timestamp check
        last_ts = data.index[-1].to_pydatetime().replace(tzinfo=pytz.UTC)
        now = datetime.now(pytz.UTC)
        
        if (now - last_ts).total_seconds() > 300:
            logger.warning(f"Stale data: {last_ts} UTC")
            return pd.DataFrame()
            
        return data
        
    except Exception as e:
        logger.error(f"Data fetch failed: {str(e)}")
        return pd.DataFrame()

def main():
    st.set_page_config(page_title="Crypto Debug", layout="centered")
    st.title("Minimal Working Version")
    
    try:
        st.write("## Data Availability Check")
        
        # Test BTC data fetch
        btc_data = get_realtime_data('BTC-USD')
        
        if not btc_data.empty:
            latest_price = btc_data['Close'].iloc[-1]
            st.success(f"Latest BTC Price: ${latest_price:.2f}")
            st.write("Raw data preview:")
            st.dataframe(btc_data.tail(3))
        else:
            st.error("No market data available - check network connection")
            
    except Exception as e:
        logger.critical(f"Main execution failed: {str(e)}")
        st.error("Application error - contact support")

if __name__ == "__main__":
    main()
