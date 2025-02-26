import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime, timedelta

# Configuration
CRYPTO_PAIRS = ['BTC-GBP', 'ETH-GBP', 'BNB-GBP', 'XRP-GBP', 'ADA-GBP']
UK_TIMEZONE = pytz.timezone('Europe/London')
PERFORMANCE_LOG = "trading_performance.csv"

@st.cache_data(ttl=300)
def get_enhanced_data(pair, period='5d', interval='15m'):
    """Improved data fetching with error recovery"""
    try:
        ticker = yf.Ticker(pair)
        data = ticker.history(period=period, interval=interval)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Data Error: {str(e)}")
        return None

def calculate_volatility(data):
    """Calculate historical volatility"""
    returns = np.log(data['Close']/data['Close'].shift(1))
    return returns.std() * np.sqrt(252)  # Annualized volatility

def calculate_advanced_indicators(data):
    """Additional technical indicators"""
    # MACD
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    
    # Stochastic Oscillator
    low14 = data['Low'].rolling(window=14).min()
    high14 = data['High'].rolling(window=14).max()
    k = 100 * ((data['Close'] - low14) / (high14 - low14))
    d = k.rolling(window=3).mean()
    
    return {
        'macd': macd.iloc[-1],
        'signal': signal.iloc[-1],
        'stochastic_k': k.iloc[-1],
        'stochastic_d': d.iloc[-1]
    }

def log_performance(pair, entry_price, exit_price, quantity):
    """Track trading performance"""
    trade_date = datetime.now(UK_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    profit = (exit_price - entry_price) * quantity
    log_entry = {
        'date': trade_date,
        'pair': pair,
        'entry': entry_price,
        'exit': exit_price,
        'quantity': quantity,
        'profit': profit
    }
    
    try:
        df = pd.DataFrame([log_entry])
        df.to_csv(PERFORMANCE_LOG, mode='a', header=not os.path.exists(PERFORMANCE_LOG), index=False)
    except Exception as e:
        st.error(f"Logging Error: {str(e)}")

def main():
    st.set_page_config(page_title="Pro Crypto Trader", layout="wide")
    
    st.title("🚀 Enhanced Crypto Trading Bot")
    st.write("### Advanced Trading Analytics")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Balance (£):", 
                                      min_value=100, 
                                      max_value=1000000, 
                                      value=1000,
                                      step=500)
        risk_percent = st.slider("Risk Percentage:", 
                                min_value=1, 
                                max_value=10, 
                                value=2,
                                help="Percentage of account to risk per trade")
    
    with col2:
        data = get_enhanced_data(pair)
        if data is not None:
            current_price = data['Close'].iloc[-1]
            volatility = calculate_volatility(data)
            
            # Enhanced Price Display
            fig_price = go.Figure()
            fig_price.add_trace(go.Candlestick(x=data.index,
                             open=data['Open'],
                             high=data['High'],
                             low=data['Low'],
                             close=data['Close'],
                             name='Price'))
            
            # Add moving averages
            fig_price.add_trace(go.Scatter(x=data.index, 
                                         y=data['Close'].rolling(20).mean(),
                                         line=dict(color='orange', width=1.5),
                                         name='20 MA'))
            
            fig_price.update_layout(title=f"{pair} Price Action",
                                  xaxis_title="Time",
                                  yaxis_title="Price (£)",
                                  height=500)
            
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Risk Management Calculator
            st.subheader("Risk Management")
            col_risk1, col_risk2 = st.columns(2)
            
            with col_risk1:
                stop_loss_pct = st.slider("Stop Loss (%)", 
                                         min_value=0.5, 
                                         max_value=10.0, 
                                         value=2.0, 
                                         step=0.5)
                
                position_size = (account_size * (risk_percent/100)) / (current_price * (stop_loss_pct/100))
                st.metric("Position Size", f"{position_size:.4f} {pair.split('-')[0]}")
            
            with col_risk2:
                take_profit_pct = st.slider("Take Profit (%)", 
                                           min_value=1.0, 
                                           max_value=20.0, 
                                           value=5.0, 
                                           step=0.5)
                st.metric("Volatility (Annualized)", f"{volatility*100:.2f}%")
            
            # Advanced Indicators
            st.subheader("Technical Analytics")
            indicators = calculate_advanced_indicators(data)
            
            col_ind1, col_ind2 = st.columns(2)
            with col_ind1:
                st.write("**MACD:**")
                st.write(f"MACD Line: {indicators['macd']:.2f}")
                st.write(f"Signal Line: {indicators['signal']:.2f}")
                
            with col_ind2:
                st.write("**Stochastic Oscillator:**")
                st.write(f"%K: {indicators['stochastic_k']:.2f}")
                st.write(f"%D: {indicators['stochastic_d']:.2f}")
            
            # Performance Tracking
            if st.button("Simulate Random Trade"):
                quantity = position_size
                exit_price = current_price * (1 + np.random.choice([-stop_loss_pct/100, take_profit_pct/100]))
                log_performance(pair, current_price, exit_price, quantity)
                st.success("Trade logged successfully!")

if __name__ == "__main__":
    main()
