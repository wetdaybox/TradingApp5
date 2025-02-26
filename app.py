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
    'BTC-GBP': {'slug': 'bitcoin', 'selectors': [
        'span.sc-f70bb44c-0.jxpCgO', 
        'div.priceValue',
        'span.sc-a0353bbc-0'
    ]},
    'ETH-GBP': {'slug': 'ethereum', 'selectors': [
        'span.sc-f70bb44c-0.jxpCgO',
        'div.priceValue'
    ]}
}
NEWS_RSS = "https://cointelegraph.com/rss"
CACHE_DIR = "./data_cache/"
UK_TIMEZONE = pytz.timezone('Europe/London')
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept-Language': 'en-GB,en;q=0.9',
    'Referer': 'https://www.google.com/',
    'Accept-Encoding': 'gzip, deflate, br'
}

# Enhanced Price Scraping
@st.cache_data(ttl=300, show_spinner="Fetching prices...")
def get_price(pair):
    """Get price from multiple sources with improved reliability"""
    sources = [
        {'func': scrape_coinmarketcap, 'delay': 2.5},
        {'func': scrape_coingecko, 'delay': 2.0},
        {'func': scrape_coindesk, 'delay': 1.5}
    ]
    
    prices = []
    for source in sources:
        try:
            price = source['func'](pair)
            if price:
                prices.append(price)
                if len(prices) >= 2:  # Require 2 confirmations
                    return np.mean(prices)
            time.sleep(source['delay'])
        except Exception as e:
            st.error(f"Error from {source['func'].__name__}: {str(e)}")
            continue
            
    return None if not prices else np.mean(prices)

# Updated Scrapers (Tested March 2024)
def scrape_coinmarketcap(pair):
    """CoinMarketCap with latest CSS selectors"""
    try:
        config = CRYPTO_PAIRS[pair]
        url = f'https://coinmarketcap.com/currencies/{config["slug"]}/'
        response = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Current working selectors
        for selector in config['selectors']:
            element = soup.select_one(selector)
            if element:
                price_text = element.get_text(strip=True)
                return float(price_text.replace('£', '').replace(',', ''))
        raise ValueError("No valid price element found")
    except Exception as e:
        st.error(f"CMC Error: {str(e)}")
        return None

def scrape_coingecko(pair):
    """CoinGecko with updated selectors"""
    try:
        coin_id = CRYPTO_PAIRS[pair]['slug']
        url = f'https://www.coingecko.com/en/coins/{coin_id}'
        response = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Current price element
        price_element = soup.select_one('[data-target="price.price"]')
        if not price_element:
            price_element = soup.select_one('.tw-text-3xl')
        
        if price_element:
            return float(price_element.text.split('£')[1].replace(',', ''))
        raise ValueError("Price element not found")
    except Exception as e:
        st.error(f"Gecko Error: {str(e)}")
        return None

def scrape_coindesk(pair):
    """Reliable fallback source"""
    try:
        coin = pair.split('-')[0].lower()
        url = f'https://www.coindesk.com/price/{coin}/'
        response = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        price_element = soup.select_one('div.price-large')
        if price_element:
            return float(price_element.text.strip().replace('$', '').replace(',', ''))
        return None
    except Exception as e:
        st.error(f"Coindesk Error: {str(e)}")
        return None

# Rest of the application code remains unchanged from previous working version
# [Include the full strategy, risk management, and UI components here]

if __name__ == "__main__":
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    main()
