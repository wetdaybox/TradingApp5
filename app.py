import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
import sqlite3
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# 🇬🇧 British Configuration 🇬🇧
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'BNB-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')
RSI_PERIOD = 14

# Database setup for trade history
conn = sqlite3.connect('trading_journal.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS trades
            (id INTEGER PRIMARY KEY,
             timestamp TEXT,
             pair TEXT,
             action TEXT,
             price REAL,
             amount REAL,
             outcome TEXT)''')
conn.commit()

# Initialize core session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now(UK_TIMEZONE).strftime("%H:%M:%S")

def get_rsi(data, window=14):
    """Original RSI calculation preserved"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=30)
def get_realtime_data(pair):
    """Original data fetching with improvements"""
    try:
        data = yf.download(pair, period='2d', interval='5m', progress=False)
        if not data.empty:
            data.index = data.index.tz_convert(UK_TIMEZONE)
            data['RSI'] = get_rsi(data)
            st.session_state.last_update = datetime.now(UK_TIMEZONE).strftime("%H:%M:%S")
        return data
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

def calculate_trading_signals(data):
    """Enhanced with professional indicators"""
    signals = {}
    
    # Original RSI logic
    signals['rsi'] = data['RSI'].iloc[-1]
    
    # Professional enhancements
    data['SMA20'] = data['Close'].rolling(20).mean()
    data['SMA50'] = data['Close'].rolling(50).mean()
    data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    
    signals.update({
        'golden_cross': data['SMA20'].iloc[-1] > data['SMA50'].iloc[-1],
        'macd_rising': data['MACD'].iloc[-1] > data['MACD'].iloc[-2]
    })
    
    return signals

def generate_trade_advice(signals, current_price):
    """Plain English explanations"""
    advice = []
    
    # RSI based
    if signals['rsi'] < 30:
        advice.append("🔥 STRONG BUY: Oversold territory (RSI < 30)")
    elif signals['rsi'] > 70:
        advice.append("💰 TAKE PROFIT: Overbought territory (RSI > 70)")
    else:
        advice.append("🟢 HOLD: Neutral RSI reading")
    
    # Trend based
    if signals['golden_cross']:
        advice.append("📈 TREND UP: 20-period average crossed above 50-period")
    if signals['macd_rising']:
        advice.append("🚀 MOMENTUM: Rising MACD indicator")
    
    # Price action
    advice.append(f"💷 Current Price: £{current_price:,.2f}")
    
    return advice

def main():
    st.set_page_config(page_title="🇬🇧 Crypto Trader Pro", layout="centered")
    st.title("🇬🇧 Personal Trading Assistant")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Asset:", CRYPTO_PAIRS)
        st.caption(f"Last update: {st.session_state.last_update}")
        
        # Trade execution
        with st.form("trade_execution"):
            trade_amount = st.number_input("Amount (£)", min_value=10.0, value=100.0)
            if st.form_submit_button("📈 Execute Buy"):
                # Store trade in database
                current_price = get_realtime_data(pair)['Close'].iloc[-1]
                c.execute('''INSERT INTO trades 
                          (timestamp, pair, action, price, amount)
                          VALUES (?,?,?,?,?)''',
                        (datetime.now(UK_TIMEZONE).isoformat(), pair, 
                         "BUY", current_price, trade_amount))
                conn.commit()
                st.success("Trade executed successfully!")
    
    with col2:
        data = get_realtime_data(pair)
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            signals = calculate_trading_signals(data)
            advice = generate_trade_advice(signals, current_price)
            
            # Display trading signals
            st.subheader("Trading Signals")
            for signal in advice:
                st.markdown(f"- {signal}")
            
            # Enhanced price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], name='20-period MA'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='50-period MA'))
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade history
            st.subheader("Recent Trades")
            trades = pd.read_sql('''SELECT timestamp, pair, action, price, amount 
                                  FROM trades ORDER BY timestamp DESC LIMIT 5''', conn)
            st.dataframe(trades.style.format({
                'price': '£{:.2f}',
                'amount': '£{:.2f}'
            }))
            
        else:
            st.warning("Loading market data...")

if __name__ == "__main__":
    main()
