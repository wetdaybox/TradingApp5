import streamlit as st
import pandas as pd
import yfinance as yf
import logging  # NEW

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)  # NEW

CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'

@st.cache_data(ttl=60)
def get_realtime_data(pair):
    try:
        logger.info(f"Fetching data for {pair}")  # NEW
        data = yf.download(pair, period='1d', interval='1m')
        logger.info(f"Received {len(data)} rows for {pair}")  # NEW
        return data
    except Exception as e:
        logger.error(f"Data error: {str(e)}")  # NEW
        return pd.DataFrame()

def get_fx_rate():
    try:
        logger.info("Fetching FX rate")  # NEW
        fx_data = yf.download(FX_PAIR, period='1d')
        rate = fx_data['Close'].iloc[-1]
        logger.info(f"FX rate: {rate}")  # NEW
        return rate
    except Exception as e:
        logger.error(f"FX error: {str(e)}")  # NEW
        return 0.80

def main():
    logger.info("App started")  # NEW
    st.set_page_config(page_title="Crypto Trader", layout="centered")
    st.title("🇬🇧 Free Crypto Trading Bot")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Balance (£):", 
                                     min_value=100, value=1000)
    
    with col2:
        logger.info(f"Processing {pair}")  # NEW
        data = get_realtime_data(pair)
        if not data.empty:
            fx_rate = get_fx_rate()
            current_price = data['Close'].iloc[-1] / fx_rate
            
            st.metric("Current Price", f"£{current_price:.2f}")
            
            high = data['High'].iloc[-20:].max()
            low = data['Low'].iloc[-20:].min()
            
            st.write(f"24h Range: £{low/fx_rate:.2f} - £{high/fx_rate:.2f}")
        else:
            logger.warning("Empty data received")  # NEW

if __name__ == "__main__":
    main()
