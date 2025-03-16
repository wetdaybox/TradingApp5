import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
from datetime import datetime, timedelta
import sqlite3
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
             (date TEXT, pair TEXT, price REAL)''')
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
        # Store in local DB
        for index, row in data.iterrows():
            c.execute("INSERT INTO price_history VALUES (?,?,?)", 
                     (index.strftime("%Y-%m-%d %H:%M"), pair, row['Close']))
        conn.commit()
    return data

def simple_predictor(data):
    model = LinearRegression()
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data['Close'].values
    model.fit(X, y)
    prediction = model.predict([[len(X)]])[0]
    return round(prediction, 2)

def calculate_signals(data):
    # Simple moving averages
    data['SMA20'] = data['Close'].rolling(20).mean()
    data['SMA50'] = data['Close'].rolling(50).mean()
    
    # MACD
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data

# Trading Strategy
def generate_advice(data, current_price):
    advice = []
    
    # Golden Cross
    if data['SMA20'].iloc[-1] > data['SMA50'].iloc[-1]:
        advice.append(("💰 Golden Cross Detected!", "20-day average crossed above 50-day"))
    
    # MACD Cross
    if data['MACD'].iloc[-1] > data['Signal'].iloc[-1]:
        advice.append(("📈 Momentum Building", "MACD crossed above signal line"))
    
    # Simple Prediction
    prediction = simple_predictor(data)
    advice.append((f"📊 Next Hour Prediction: £{prediction}", 
                  f"Based on recent price movement (current: £{current_price})"))
    
    return advice

# Main App
def main():
    st.set_page_config(page_title="🇬🇧 Crypto Trader Pro", layout="centered")
    st.title("🇬🇧 Your Personal Crypto Trading Partner")
    
    with st.expander("📘 How to Use This Assistant"):
        st.write("""
        1. **Choose** your cryptocurrency
        2. See **clear buy/sell signals**
        3. Follow the **simple instructions**
        4. Track your **virtual trades**
        """)
    
    pair = st.selectbox("1. Choose Cryptocurrency:", CRYPTO_PAIRS)
    
    # Get data
    data = get_historical_data(pair)
    current_price = data['Close'].iloc[-1] if not data.empty else None
    
    if current_price:
        data = calculate_signals(data)
        advice = generate_advice(data, current_price)
        
        # Display Signals
        st.subheader("🚦 Trading Signals")
        for signal, explanation in advice:
            st.success(f"{signal} - {explanation}")
        
        # Action Buttons
        st.subheader("📋 Your Trading Plan")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💷 Buy Now"):
                c.execute("INSERT INTO trades VALUES (?,?,?,?,?)",
                          (datetime.now().strftime("%Y-%m-%d %H:%M"), pair, "BUY", current_price, 100))
                conn.commit()
                st.session_state.last_action = f"Bought £100 of {pair} at £{current_price}"
        
        with col2:
            if st.button("💵 Take Profit"):
                c.execute("INSERT INTO trades VALUES (?,?,?,?,?)",
                          (datetime.now().strftime("%Y-%m-%d %H:%M"), pair, "SELL", current_price, 100))
                conn.commit()
                st.session_state.last_action = f"Sold £100 of {pair} at £{current_price}"
        
        with col3:
            st.metric("💷 Current Price", f"£{current_price:,.2f}")
        
        # Trading History
        st.subheader("📜 Your Trading Journal")
        trades = pd.read_sql("SELECT * FROM trades ORDER BY date DESC LIMIT 5", conn)
        st.table(trades.style.applymap(lambda x: f"color: {get_color(x=='BUY')}", subset=['action']))
        
        # Price Chart
        st.subheader("📈 Price History")
        fig = go.Figure(data=[
            go.Scatter(x=data.index, y=data['Close'], name='Price'),
            go.Scatter(x=data.index, y=data['SMA20'], name='20-hr Average'),
            go.Scatter(x=data.index, y=data['SMA50'], name='50-hr Average')
        ])
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"Last action: {st.session_state.last_action}")
    else:
        st.warning("Loading market data... Please wait a moment!")

if __name__ == "__main__":
    main()
