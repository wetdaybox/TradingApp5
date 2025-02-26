# trading_bot.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from bs4 import BeautifulSoup
import requests
import time
import feedparser
import os

# Configuration
CRYPTO_PAIRS = {
    'BTC-GBP': {'slug': 'bitcoin', 'selector': 'div.priceValue > span'},
    'ETH-GBP': {'slug': 'ethereum', 'selector': 'div.priceValue > span'},
    'BNB-GBP': {'slug': 'bnb', 'selector': 'div.priceValue > span'},
    'XRP-GBP': {'slug': 'ripple', 'selector': 'div.priceValue > span'},
    'ADA-GBP': {'slug': 'cardano', 'selector': 'div.priceValue > span'}
}
NEWS_RSS = "https://cointelegraph.com/rss"
CACHE_DIR = "./data_cache/"
PERFORMANCE_LOG = os.path.join(CACHE_DIR, "performance_log.csv")
UK_TIMEZONE = pytz.timezone('Europe/London')
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept-Language': 'en-GB,en;q=0.9',
    'Referer': 'https://www.google.com/'
}

# ----------------------
# Enhanced Data Layer
# ----------------------
@st.cache_data(ttl=300, show_spinner="Fetching prices...")
def get_multisource_price(pair):
    """Get price from 2 sources with error recovery"""
    try:
        price1 = scrape_coinmarketcap(pair)
        time.sleep(2.5)  # Increased delay to avoid blocking
        price2 = scrape_coingecko(pair)
        
        if price1 and price2:
            return round((price1 + price2)/2, 2)
        return price1 or price2
    except Exception as e:
        st.error(f"Price fetch failed: {str(e)}")
        return None

def scrape_coinmarketcap(pair):
    """Updated March 2024 CSS selectors"""
    try:
        config = CRYPTO_PAIRS[pair]
        url = f'https://coinmarketcap.com/currencies/{config["slug"]}/'
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Multiple fallback selectors
        price_element = (
            soup.select_one('span.sc-f70bb44c-0.jxpCgO.base-text') or
            soup.select_one('div.priceValue > span') or
            soup.select_one('.sc-a0353bbc-0.gDrtaY')
        )
        
        if not price_element:
            raise ValueError("Price element not found on CoinMarketCap")
            
        return float(price_element.text.strip('£').replace(',', ''))
    except Exception as e:
        st.error(f"CMC Error: {str(e)}")
        return None

def scrape_coingecko(pair):
    """Updated March 2024 selectors"""
    try:
        coin_id = CRYPTO_PAIRS[pair]['slug']
        url = f'https://www.coingecko.com/en/coins/{coin_id}'
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Multiple fallback selectors
        price_element = (
            soup.select_one('[data-target="price.price"]') or
            soup.select_one('.tw-text-3xl') or
            soup.select_one('.no-wrap')
        )
        
        if not price_element:
            raise ValueError("Price element not found on CoinGecko")
            
        price_text = price_element.text.strip()
        if '£' in price_text:
            return float(price_text.split('£')[1].replace(',', ''))
        return float(price_text.replace(',', ''))
    except Exception as e:
        st.error(f"Coingecko Error: {str(e)}")
        return None

# ----------------------
# Strategy & Risk Management (Remaining unchanged from previous working version)
# ----------------------
# [Include all strategy functions from previous answer here]
# ----------------------
# Performance Monitoring (Remaining unchanged)
# ----------------------
# [Include all performance functions from previous answer here]
# ----------------------
# Main Application (Remaining unchanged)
# ----------------------
# [Include main() function from previous answer here]

if __name__ == "__main__":
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    main()
