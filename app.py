import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
import numpy as np
import json
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
BASE_REFRESH_INTERVAL = 60  # Seconds

# Initialize session state
if 'manual_price' not in st.session_state:
    st.session_state.manual_price = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")

@st.cache_data(ttl=30)
def get_realtime_data(pair):
    """Get 48 hours of 5-minute data"""
    try:
        data = yf.download(pair, period='2d', interval='5m', progress=False)
        if not data.empty:
            data = preprocess_data(data)
            st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        return data
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def preprocess_data(data):
    """Add technical indicators"""
    data['RSI'] = get_rsi(data)
    data = calculate_technical_indicators(data)
    return data

def get_rsi(data, window=14):
    """Enhanced RSI calculation"""
    if len(data) < window + 1:
        return pd.Series([None]*len(data))
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=60)
def get_fx_rate():
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='5m')
        return fx_data['Close'].iloc[-1].item() if not fx_data.empty else 0.80
    except Exception as e:
        st.error(f"FX error: {str(e)}")
        return 0.80

def get_price_data(pair):
    data = get_realtime_data(pair)
    fx_rate = get_fx_rate()
    
    if st.session_state.manual_price is not None:
        return st.session_state.manual_price, True
    
    if not data.empty:
        return data['Close'].iloc[-1].item() / fx_rate, False
    return None, False

def calculate_technical_indicators(data):
    """Calculate Bollinger Bands and MACD"""
    # Bollinger Bands
    data['20ma'] = data['Close'].rolling(20).mean()
    data['upper_band'] = data['20ma'] + 2*data['Close'].rolling(20).std()
    data['lower_band'] = data['20ma'] - 2*data['Close'].rolling(20).std()
    
    # MACD
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data

def get_market_sentiment(pair):
    """Fetch recent news headlines"""
    try:
        ticker = yf.Ticker(pair.split('-')[0])
        news = ticker.news
        return [item['title'] for item in news][:3] if news else []
    except Exception as e:
        return ["Sentiment data unavailable"]

def calculate_risk_metrics(data):
    """Calculate Sharpe ratio and drawdown"""
    returns = data['Close'].pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    max_drawdown = (data['Close'] / data['Close'].cummax() - 1).min()
    return {
        'sharpe': round(sharpe_ratio, 2),
        'drawdown': round(max_drawdown*100, 1)
    }

def main():
    st.set_page_config(page_title="Crypto Trader Pro+", layout="wide")
    st.title("🚀 Enhanced Crypto Trading Assistant")
    
    # Load user preferences
    user_prefs = load_user_prefs()
    
    with st.sidebar:
        st.header("⚙️ Settings")
        pair = st.selectbox("Select Asset:", CRYPTO_PAIRS)
        timeframe = st.selectbox("Chart Timeframe", ['5m', '15m', '30m', '1h'])
        indicators = st.multiselect("Technical Indicators", 
                                  ['Bollinger Bands', 'MACD'],
                                  default=user_prefs.get('indicators', []))
        
        st.header("📈 Risk Parameters")
        tp_percent = st.slider("Take Profit %", 1.0, 30.0, 15.0)
        sl_percent = st.slider("Stop Loss %", 1.0, 10.0, 5.0)
        risk_reward = st.slider("Risk/Reward Ratio", 1.0, 5.0, 3.0, 0.5)
    
    # Main display columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("💰 Portfolio Manager")
        account_size = st.number_input("Portfolio Value (£)", 
                                     min_value=100.0, value=1000.0, step=100.0)
        use_manual = st.checkbox("Enter Price Manually")
        if use_manual:
            st.session_state.manual_price = st.number_input(
                "Manual Price (£)", min_value=0.01, 
                value=st.session_state.manual_price or 1000.0
            )
        else:
            st.session_state.manual_price = None
            
        # Market Sentiment
        st.subheader("📰 Market Sentiment")
        for headline in get_market_sentiment(pair):
            st.markdown(f"- {headline}")
    
    with col2:
        # Real-time data display
        st.header("📊 Market Analysis")
        update_diff = (datetime.now() - datetime.strptime(st.session_state.last_update, "%H:%M:%S")).seconds
        recency_color = "green" if update_diff < 120 else "orange" if update_diff < 300 else "red"
        st.markdown(f"🕒 Last update: <span style='color:{recency_color}'>{st.session_state.last_update}</span>",
                  unsafe_allow_html=True)
        
        current_price, is_manual = get_price_data(pair)
        data = get_realtime_data(pair)
        
        if current_price and not data.empty:
            levels = calculate_levels(data, current_price, tp_percent, sl_percent)
            risk_metrics = calculate_risk_metrics(data)
            
            # Display metrics
            metric_cols = st.columns(3)
            rsi_color = "green" if levels['rsi'] < RSI_OVERSOLD else "red" if levels['rsi'] > RSI_OVERBOUGHT else "gray"
            metric_cols[0].markdown(f"<div style='text-align: center;'><h3 style='color:{rsi_color}'>{levels['rsi']}</h3>RSI</div>",
                                  unsafe_allow_html=True)
            metric_cols[1].metric("24h Range", f"£{levels['low']:,.2f}-£{levels['high']:,.2f}")
            metric_cols[2].metric("Volatility", f"£{levels['volatility']:,.2f}")
            
            # Strategy display
            with st.expander("📈 Trading Strategy Dashboard", expanded=True):
                # Price chart
                fig = go.Figure()
                resampled_data = resample_data(data, timeframe)
                
                # Candlestick plot
                fig.add_trace(go.Candlestick(
                    x=resampled_data.index,
                    open=resampled_data['Open'],
                    high=resampled_data['High'],
                    low=resampled_data['Low'],
                    close=resampled_data['Close'],
                    name='Price'
                ))
                
                # Technical indicators
                if 'Bollinger Bands' in indicators:
                    fig.add_trace(go.Scatter(
                        x=data.index, y=data['upper_band'],
                        name='Upper Band', line=dict(color='rgba(255,0,0,0.5)')
                    ))
                    fig.add_trace(go.Scatter(
                        x=data.index, y=data['lower_band'],
                        name='Lower Band', line=dict(color='rgba(0,255,0,0.5)')
                    ))
                
                if 'MACD' in indicators:
                    fig.add_trace(go.Scatter(
                        x=data.index, y=data['MACD'],
                        name='MACD', line=dict(color='blue'),
                        row=2, col=1
                    ))
                    fig.add_trace(go.Scatter(
                        x=data.index, y=data['Signal'],
                        name='Signal', line=dict(color='orange'),
                        row=2, col=1
                    ))
                
                # Strategy levels
                fig.add_hline(y=levels['buy_zone'], line_dash="dot",
                            annotation_text="Buy Zone", line_color="green")
                fig.add_hline(y=levels['take_profit'], line_dash="dot",
                            annotation_text="Take Profit", line_color="blue")
                fig.add_hline(y=levels['stop_loss'], line_dash="dot",
                            annotation_text="Stop Loss", line_color="red")
                
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Strategy signals
                signals = backtest_strategy(data)
                st.write(f"**Strategy Performance:** {signals['total_return']}%")
            
            # Risk management
            with st.expander("🛡️ Risk Management"):
                cols = st.columns(3)
                cols[0].metric("Sharpe Ratio", risk_metrics['sharpe'])
                cols[1].metric("Max Drawdown", f"{risk_metrics['drawdown']}%")
                cols[2].metric("Win Rate", f"{signals['win_rate']}%")
                
                # Position sizing
                risk_amount = st.slider("Risk Amount (£)", 10.0, account_size, 100.0)
                position_size = calculate_position_size(risk_amount, current_price, levels)
                st.write(f"""
                **Recommended Position:**
                - Size: {position_size:.4f} {pair.split('-')[0]}
                - Value: £{(position_size * current_price):,.2f}
                - Risk/Reward: 1:{risk_reward}
                """)
                
                # Scenario analysis
                scenario_analysis(current_price, levels)
    
    # Adaptive refresh
    volatility = data['Close'].pct_change().std()
    refresh_rate = BASE_REFRESH_INTERVAL - int(volatility * 1000)
    st_autorefresh(interval=max(refresh_rate, 15)*1000, key="auto_refresh")

def calculate_levels(data, current_price, tp_percent, sl_percent):
    """Calculate trading levels with volatility"""
    try:
        fx_rate = get_fx_rate()
        full_day_data = data.iloc[-288:]  # 24h data
        
        # Volatility (ATR)
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        return {
            'buy_zone': round(full_day_data['Low'].min() * 0.98 / fx_rate, 2),
            'take_profit': round(current_price * (1 + tp_percent/100), 2),
            'stop_loss': round(current_price * (1 - sl_percent/100), 2),
            'rsi': round(data['RSI'].iloc[-1], 1),
            'high': round(full_day_data['High'].max() / fx_rate, 2),
            'low': round(full_day_data['Low'].min() / fx_rate, 2),
            'volatility': round(atr / fx_rate, 2)
        }
    except Exception as e:
        st.error(f"Level calculation error: {str(e)}")
        return None

def backtest_strategy(data):
    """Backtest RSI strategy"""
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['rsi'] = data['RSI']
    signals['signal'] = 0
    
    # Generate signals
    signals['long'] = (signals['rsi'] < RSI_OVERSOLD).astype(int)
    signals['short'] = (signals['rsi'] > RSI_OVERBOUGHT).astype(int)
    
    # Calculate returns
    signals['returns'] = signals['price'].pct_change()
    signals['strategy'] = signals['long'].shift(1) * signals['returns']
    
    # Performance metrics
    total_return = signals['strategy'].dropna().cumsum().iloc[-1] * 100
    win_rate = (signals['strategy'] > 0).mean() * 100
    
    return {
        'total_return': round(total_return, 1),
        'win_rate': round(win_rate, 1)
    }

def scenario_analysis(current_price, levels):
    """Interactive scenario planner"""
    scenario = st.slider("Price Change Scenario (%)", -20, 20, 0)
    new_price = current_price * (1 + scenario/100)
    pl_ratio = (new_price - current_price) / (current_price - levels['stop_loss'])
    
    st.write(f"""
    **Scenario Analysis ({scenario}%):**
    - New Price: £{new_price:.2f}
    - P/L Ratio: {pl_ratio:.1f}:1
    - Account Impact: {min(max(pl_ratio*100, -100), 100):.1f}%
    """)

def calculate_position_size(risk_amount, current_price, levels):
    """Volatility-adjusted position sizing"""
    risk_per_unit = abs(current_price - levels['stop_loss'])
    return risk_amount / (risk_per_unit * (1 + levels['volatility']/100))

def resample_data(data, timeframe):
    """Resample data for different timeframes"""
    return data.resample(timeframe).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()

def load_user_prefs():
    """Load user preferences from file"""
    try:
        with open('user_prefs.json') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_user_prefs(prefs):
    """Save user preferences to file"""
    with open('user_prefs.json', 'w') as f:
        json.dump(prefs, f)

if __name__ == "__main__":
    main()
