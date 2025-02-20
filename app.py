import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import asyncio
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import plotly.graph_objects as go

# --- Configuration ---
SYMBOL = 'BTC-USD'
TIMEFRAME = '5m'
RISK_PARAMS = {
    'stop_loss_pct': 2.0,
    'take_profit_pct': 4.0,
    'max_position': 0.1
}

# --- Helper Functions ---

def fetch_historical_data(symbol, period='5d', interval=TIMEFRAME):
    """
    Fetch historical data using yf.Ticker().history().
    If no data is returned for the specified period, try a fallback period.
    """
    ticker = yf.Ticker(symbol)
    try:
        data = ticker.history(period=period, interval=interval, progress=False)
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        data = pd.DataFrame()
    if data.empty:
        st.warning(f"No data returned for period '{period}'. Trying fallback period '1mo'.")
        try:
            data = ticker.history(period='1mo', interval=interval, progress=False)
        except Exception as e:
            st.error(f"Error fetching fallback historical data: {e}")
            data = pd.DataFrame()
    return data

def calculate_indicators(df):
    """
    Calculate RSI (14-period) and Bollinger Bands (20-period, 2 std dev) using the 'Close' prices.
    """
    df = df.copy()
    if 'Close' not in df.columns or df.empty:
        return df
    try:
        df['rsi'] = RSIIndicator(df['Close'], window=14).rsi()
        bb = BollingerBands(df['Close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
    return df

def generate_signal(latest):
    """
    Generate a trading signal based on the latest data:
      - BUY if price < lower band and RSI < 35,
      - SELL if price > upper band and RSI > 70,
      - Otherwise HOLD.
    """
    signal = {
        'timestamp': datetime.now(),
        'price': latest.Close,
        'action': 'HOLD'
    }
    if latest.Close < latest.bb_lower and latest.rsi < 35:
        signal.update({
            'action': 'BUY',
            'stop_loss': latest.Close * (1 - RISK_PARAMS['stop_loss_pct'] / 100),
            'take_profit': latest.Close * (1 + RISK_PARAMS['take_profit_pct'] / 100)
        })
    elif latest.Close > latest.bb_upper and latest.rsi > 70:
        signal['action'] = 'SELL'
    return signal

async def get_realtime_price():
    """
    Continuously fetch the latest 1-minute price data for SYMBOL.
    If the returned data is empty, wait and retry.
    """
    ticker = yf.Ticker(SYMBOL)
    while True:
        try:
            data = ticker.history(period='1d', interval='1m', progress=False)
            if data.empty:
                st.warning("Live data empty. Retrying in 30 seconds...")
                await asyncio.sleep(30)
                continue
            yield data.iloc[-1]
            await asyncio.sleep(30)
        except Exception as e:
            st.error(f"Live data error: {e}")
            await asyncio.sleep(10)

def build_chart(df):
    """
    Build and return a Plotly candlestick chart with Bollinger Bands overlaid.
    """
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    )])
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['bb_upper'],
        line=dict(color='red'),
        name='Upper Band'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['bb_lower'],
        line=dict(color='green'),
        name='Lower Band'
    ))
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig

# --- Main Async Function ---

async def main():
    st.set_page_config(page_title="Real-Time Crypto Trading Signals", layout="wide")
    st.title("💰 Real-Time Crypto Trading Signals")
    
    # Create UI placeholders
    price_placeholder = st.empty()
    chart_placeholder = st.empty()
    signal_placeholder = st.empty()
    
    # Fetch initial historical data
    hist_data = fetch_historical_data(SYMBOL, period='5d', interval=TIMEFRAME)
    if hist_data.empty:
        st.error("Failed to download historical data for BTC-USD. Please check your network and try again.")
        return
    hist_data = calculate_indicators(hist_data)
    
    # Display the initial chart
    chart_placeholder.plotly_chart(build_chart(hist_data), use_container_width=True)
    
    # Continuously update with live data
    async for live in get_realtime_price():
        new_row = live.to_frame().T
        hist_data = pd.concat([hist_data, new_row])
        hist_data = hist_data.iloc[-100:]  # Keep the most recent 100 rows
        hist_data = calculate_indicators(hist_data)
        
        latest = hist_data.iloc[-1]
        signal = generate_signal(latest)
        
        # Calculate price change from previous data point
        if len(hist_data) > 1:
            price_change = latest.Close - hist_data.iloc[-2].Close
        else:
            price_change = 0.0
        
        price_placeholder.metric("Current Price", f"${latest.Close:.2f}", f"{price_change:+.2f}")
        chart_placeholder.plotly_chart(build_chart(hist_data), use_container_width=True)
        
        # Display the trading signal if it's BUY or SELL; otherwise show info
        if signal['action'] != 'HOLD':
            msg = f"🚨 {signal['action']} signal at {signal['timestamp'].strftime('%H:%M:%S')}\n"
            msg += f"Price: ${signal['price']:.2f}\n"
            if signal['action'] == 'BUY':
                msg += f"Stop Loss: ${signal.get('stop_loss', 0):.2f}\nTake Profit: ${signal.get('take_profit', 0):.2f}"
            signal_placeholder.success(msg)
        else:
            signal_placeholder.info("Monitoring market – no signal detected.")

if __name__ == "__main__":
    asyncio.run(main())
