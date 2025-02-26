# app.py
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
    'BTC-GBP': {'slug': 'bitcoin', 'selectors': ['span.sc-f70bb44c-0', 'div.priceValue']},
    'ETH-GBP': {'slug': 'ethereum', 'selectors': ['span.sc-f70bb44c-0', 'div.priceValue']},
    'BNB-GBP': {'slug': 'bnb', 'selectors': ['span.sc-f70bb44c-0', 'div.priceValue']},
    'XRP-GBP': {'slug': 'ripple', 'selectors': ['span.sc-f70bb44c-0', 'div.priceValue']},
    'ADA-GBP': {'slug': 'cardano', 'selectors': ['span.sc-f70bb44c-0', 'div.priceValue']}
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
    """Get price from multiple sources with fallback"""
    sources = [
        lambda: scrape_coinmarketcap(pair),
        lambda: scrape_coingecko(pair),
        lambda: scrape_coindesk(pair)
    ]
    
    prices = []
    for source in sources:
        try:
            price = source()
            if price:
                prices.append(price)
                if len(prices) >= 2:  # Get at least 2 successful prices
                    break
            time.sleep(1.5)
        except Exception as e:
            continue
            
    return np.mean(prices) if prices else None

def scrape_coinmarketcap(pair):
    """Updated CSS selectors for CoinMarketCap 2024 layout"""
    try:
        config = CRYPTO_PAIRS[pair]
        url = f'https://coinmarketcap.com/currencies/{config["slug"]}/'
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Multiple fallback selectors
        for selector in config['selectors'] + ['span.sc-f70bb44c-0.jxpCgO']:
            price_element = soup.select_one(selector)
            if price_element:
                price_text = price_element.get_text(strip=True)
                return float(price_text.strip('£').replace(',', ''))
                
        raise ValueError("No valid price element found")
        
    except Exception as e:
        st.error(f"CMC Error: {str(e)}")
        return None

def scrape_coingecko(pair):
    """Updated selectors for CoinGecko 2024 layout"""
    try:
        coin_id = CRYPTO_PAIRS[pair]['slug']
        url = f'https://www.coingecko.com/en/coins/{coin_id}'
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Multiple fallback selectors
        selectors = [
            '[data-target="price.price"]',
            '.tw-text-3xl',
            '.no-wrap',
            'span[data-coin-symbol]'
        ]
        
        for selector in selectors:
            price_element = soup.select_one(selector)
            if price_element:
                price_text = price_element.get_text(strip=True)
                if '£' in price_text:
                    return float(price_text.split('£')[1].replace(',', ''))
                return float(price_text.replace(',', ''))
                
        raise ValueError("No valid price element found")
        
    except Exception as e:
        st.error(f"Coingecko Error: {str(e)}")
        return None

def scrape_coindesk(pair):
    """Fallback price source"""
    try:
        coin_symbol = pair.split('-')[0]
        url = f'https://www.coindesk.com/price/{coin_symbol}-gbp/'
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        price_element = soup.select_one('div.price-large')
        if price_element:
            return float(price_element.text.strip().replace('£', '').replace(',', ''))
        return None
    except:
        return None

# ----------------------
# Remaining Strategy and UI Code (Unchanged from previous working version)
# ----------------------
# [Include all other functions from previous answer here]

if __name__ == "__main__":
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    main()
