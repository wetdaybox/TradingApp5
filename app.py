import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pytz
from datetime import datetime

# Configuration
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
RSI_OVERSOLD = 30  # Buy signal threshold
RSI_OVERBOUGHT = 70  # Take profit signal threshold

# Session state initialization
if 'manual_price' not in st.session_state:
    st.session_state.manual_price = None

def get_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=60)
def get_realtime_data(pair):
    """Enhanced data fetch with RSI calculation"""
    try:
        data = yf.download(pair, period='2d', interval='15m', progress=False)
        if not data.empty:
            data['RSI'] = get_rsi(data)
        return data
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_fx_rate():
    """Get GBP/USD rate with auto-refresh"""
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='5m')
        return fx_data['Close'].iloc[-1].item() if not fx_data.empty else 0.80
    except Exception as e:
        st.error(f"FX error: {str(e)}")
        return 0.80

def get_price_data(pair):
    """Get price with manual override option"""
    data = get_realtime_data(pair)
    fx_rate = get_fx_rate()
    
    if st.session_state.manual_price is not None:
        return st.session_state.manual_price, True
    
    if not data.empty:
        return data['Close'].iloc[-1].item() / fx_rate, False
    return None, False

def calculate_levels(pair, current_price):
    """Dynamic level calculation with RSI signals"""
    data = get_realtime_data(pair)
    if data.empty or len(data) < 20:
        return None
    
    try:
        # Calculate dynamic support/resistance
        recent_low = data['Low'].iloc[-20:].min().item()
        recent_high = data['High'].iloc[-20:].max().item()
        fx_rate = get_fx_rate()
        
        # RSI-based signals
        last_rsi = data['RSI'].iloc[-1]
        buy_zone = recent_low * 0.98 / fx_rate  # 2% below recent low
        take_profit = current_price * 1.15  # 15% profit target
        stop_loss = current_price * 0.95  # 5% stop loss
        
        return {
            'buy_zone': round(buy_zone, 2),
            'take_profit': round(take_profit, 2),
            'stop_loss': round(stop_loss, 2),
            'rsi': round(last_rsi, 1)
        }
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Crypto Trader Pro", layout="centered")
    st.title("🔮 Smart Crypto Trading Assistant")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Asset:", CRYPTO_PAIRS)
        
        # Manual price override
        use_manual = st.checkbox("Enter Price Manually")
        if use_manual:
            st.session_state.manual_price = st.number_input(
                "Manual Price (£)", min_value=0.01, value=st.session_state.manual_price or 1000.0
            )
        else:
            st.session_state.manual_price = None
            
        account_size = st.number_input("Portfolio Value (£)", 
                                     min_value=100.0, value=1000.0, step=100.0)
        
        # Risk management
        risk_profile = st.select_slider("Risk Profile:", 
                                      options=['Safety First', 'Balanced', 'High Risk'])
        
    with col2:
        current_price, is_manual = get_price_data(pair)
        if current_price:
            levels = calculate_levels(pair, current_price)
            if levels:
                # Dynamic buy/take profit signals
                buy_signal = levels['rsi'] < RSI_OVERSOLD
                take_profit_signal = levels['rsi'] > RSI_OVERBOUGHT
                
                # Visual alerts
                alert_cols = st.columns(3)
                alert_cols[0].metric("RSI", f"{levels['rsi']}",
                                   help="30=Oversold (Buy Signal), 70=Overbought (Take Profit)",
                                   delta="🔥 Buy Now!" if buy_signal else None)
                alert_cols[1].metric("Current Price", f"£{current_price:,.2f}",
                                   delta="Manual Input" if is_manual else "Live Data")
                alert_cols[2].metric("Next Target", f"£{levels['take_profit']:,.2f}",
                                   delta="💰 Take Profit" if take_profit_signal else None)
                
                # Trading signals with explanations
                with st.expander("Trading Plan"):
                    st.write(f"""
                    **Recommended Action:**  
                    { 'Consider buying - RSI indicates oversold' if buy_signal else 
                     'Consider taking profits - RSI indicates overbought' if take_profit_signal else 
                     'Hold - Waiting for stronger signals'}
                     
                    **Entry Zone:** £{levels['buy_zone']:,.2f}  
                    **Profit Target:** £{levels['take_profit']:,.2f} (+15%)  
                    **Stop Loss:** £{levels['stop_loss']:,.2f} (-5%)
                    """)
                    
                    # Interactive chart
                    fig = go.Figure(data=[
                        go.Scatter(
                            x=pd.date_range(end=datetime.now(), periods=len(get_realtime_data(pair)), freq='15T'),
                            y=get_realtime_data(pair)['Close'],
                            name='Price'
                        ),
                        go.Scatter(
                            x=[datetime.now()],
                            y=[current_price],
                            mode='markers',
                            marker=dict(color='red', size=10),
                            name='Current Price'
                        )
                    ])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk management calculator
                st.write("## Position Builder")
                risk_amount = st.slider("Risk Amount (£)", 10.0, account_size, 100.0)
                position_size = risk_amount / (current_price - levels['stop_loss'])
                st.write(f"""
                **Suggested Position:**  
                - Size: {position_size:.4f} {pair.split('-')[0]}  
                - Value: £{(position_size * current_price):,.2f}  
                - Risk/Reward Ratio: 1:3
                """)
                
            else:
                st.error("Market data unavailable")
        else:
            st.warning("Waiting for price data...")

if __name__ == "__main__":
    main()
