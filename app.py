import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
from datetime import datetime, timedelta
import sqlite3
import numpy as np
from sklearn.linear_model import LinearRegression

# 🇬🇧 British Configuration 🇬🇧
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'BNB-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Set up local database
conn = sqlite3.connect('trading_journal.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS trades
             (date TEXT, pair TEXT, action TEXT, price REAL, amount REAL)''')
c.execute('''CREATE TABLE IF NOT EXISTS price_history
             (date TEXT PRIMARY KEY, pair TEXT, price REAL)''')
conn.commit()

# Initialize session state
if 'last_action' not in st.session_state:
    st.session_state.last_action = "No trades yet!"

# Helper functions
def get_color(condition):
    return "green" if condition else "red"

@st.cache_data(ttl=30)
def get_historical_data(pair):
    data = yf.download(pair, period='1mo', interval='1h', progress=False, auto_adjust=True)
    if not data.empty:
        data.index = data.index.tz_convert(UK_TIMEZONE)
        # Convert to native Python types for SQLite
        records = [
            (index.strftime("%Y-%m-%d %H:%M"), pair, float(row['Close']))
            for index, row in data.iterrows()
        ]
        # Bulk insert with proper data types
        c.executemany("INSERT OR IGNORE INTO price_history VALUES (?,?,?)", records)
        conn.commit()
    return data

def simple_predictor(data):
    model = LinearRegression()
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data['Close'].values.astype(float)
    model.fit(X, y)
    prediction = model.predict([[len(X)]])[0]
    return round(prediction, 2)

# Rest of the code remains the same...

def main():
    st.set_page_config(page_title="🇬🇧 Crypto Trader Pro", layout="centered")
    st.title("🇬🇧 Your Personal Crypto Trading Partner")
    
    # ... [Keep the rest of the main() implementation unchanged]

if __name__ == "__main__":
    main()
