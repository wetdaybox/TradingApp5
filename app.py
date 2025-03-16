import streamlit as st
import pandas as pd
import yfinance as yf
import logging
import pytz
from datetime import datetime

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
FX_MIN = 1.20
FX_MAX = 1.40
DEFAULT_FX = 1.25

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_data(ttl=60)
def get_realtime_data(pair):
    """Safe data fetching with error handling"""
    try:
        logger.info(f"Fetching data for {pair}")
        data = yf.download(pair, period='1d', interval='1m', progress=False)
        if not data.empty:
            logger.info(f"Received {len(data)} rows")
            return data
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Data error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_fx_rate():
    """Robust FX rate handling with validation"""
    try:
        logger.info("Fetching FX rate")
        fx_data = yf.download(FX_PAIR, period='1d', progress=False)
        
        if not fx_data.empty:
            rate = fx_data['Close'].iloc[-1].item()  # Critical fix: .item()
            logger.info(f"Raw FX rate: {rate}")
            
            if FX_MIN <= rate <= FX_MAX:
                return rate
            logger.warning(f"FX out of bounds: {rate}")
        
        logger.warning("Using default FX rate")
        return DEFAULT_FX
        
    except Exception as e:
        logger.error(f"FX error: {str(e)}")
        return DEFAULT_FX

def main():
    logger.info("App initialized")
    st.set_page_config(page_title="Crypto Trader", layout="centered")
    st.title("🇬🇧 Free Crypto Trading Bot")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Balance (£):", 
                                     min_value=100, value=1000)
    
    with col2:
        data = get_realtime_data(pair)
        if not data.empty:
            try:
                fx_rate = get_fx_rate()
                current_price = data['Close'].iloc[-1].item() / fx_rate  # Critical fix: .item()
                
                st.metric("Current Price", f"£{current_price:,.2f}")
                
                high = data['High'].iloc[-20:].max().item()
                low = data['Low'].iloc[-20:].min().item()
                
                st.write(f"24h Range: £{low/fx_rate:,.2f} - £{high/fx_rate:,.2f}")
                
            except Exception as e:
                logger.error(f"Display error: {str(e)}")
                st.error("Temporary data issue - refresh to retry")
        else:
            st.error("Market data unavailable - try again later")

if __name__ == "__main__":
    main()
