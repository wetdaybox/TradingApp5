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
RISK_STRATEGIES = {
    'Conservative': {'stop_loss_pct': 0.15, 'take_profit_pct': 0.25},
    'Moderate': {'stop_loss_pct': 0.25, 'take_profit_pct': 0.50},
    'Aggressive': {'stop_loss_pct': 0.35, 'take_profit_pct': 0.75}
}

@st.cache_data(ttl=60)
def get_realtime_data(pair):
    """Fetch real-time crypto prices with error handling"""
    try:
        return yf.download(pair, period='1d', interval='1m', progress=False)
    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_fx_rate():
    """Get GBP/USD exchange rate with fallback"""
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='1m')
        return fx_data['Close'].iloc[-1].item() if not fx_data.empty else 0.80
    except Exception as e:
        st.error(f"FX error: {str(e)}")
        return 0.80

def get_current_price(pair):
    """Convert USD price to GBP"""
    data = get_realtime_data(pair)
    fx_rate = get_fx_rate()
    if not data.empty:
        usd_price = data['Close'].iloc[-1].item()
        return round(usd_price / fx_rate, 2)
    return None

def calculate_levels(pair):
    """Calculate trading levels based on selected strategy"""
    data = get_realtime_data(pair)
    if data.empty or len(data) < 20:
        return None
    
    try:
        strategy = st.session_state.selected_strategy
        params = RISK_STRATEGIES[strategy]
        
        closed_data = data.iloc[:-1] if len(data) > 1 else data
        high = closed_data['High'].iloc[-20:].max().item()
        low = closed_data['Low'].iloc[-20:].min().item()
        fx_rate = get_fx_rate()
        
        buy_zone = (high + low) / 2 / fx_rate
        current_price = data['Close'].iloc[-1].item() / fx_rate
        
        return {
            'buy_zone': round(buy_zone, 2),
            'take_profit': round(buy_zone * (1 + params['take_profit_pct']), 2),
            'stop_loss': round(buy_zone * (1 - params['stop_loss_pct']), 2),
            'current': round(current_price, 2)
        }
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        return None

def calculate_position_size(account_size, risk_percent, stop_loss_distance):
    """Risk-managed position sizing"""
    try:
        stop_loss_distance = max(0.0001, float(stop_loss_distance))
        risk_amount = account_size * (risk_percent / 100)
        position_size = min(risk_amount / stop_loss_distance, account_size * 2)
        return round(position_size, 4)
    except Exception as e:
        st.error(f"Position error: {str(e)}")
        return 0.0

def main():
    st.set_page_config(page_title="Crypto Trader", layout="centered")
    st.title("🇬🇧 Free Crypto Trading Bot")
    st.write("### Strategy-Driven Trading Signals")
    
    # Initialize session state for strategy persistence
    if 'selected_strategy' not in st.session_state:
        st.session_state.selected_strategy = 'Conservative'
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Balance (£)", 
                                     min_value=100.0, max_value=1000000.0, 
                                     value=1000.0, step=100.0)
        st.session_state.selected_strategy = st.selectbox(
            "Trading Strategy:", list(RISK_STRATEGIES.keys())
        )
        risk_percent = st.slider("Risk Percentage:", 1, 10, 2)
        risk_color = "#FF4B4B" if risk_percent > 5 else "#00CC96"
        st.markdown(f"<div style='background:{risk_color}; padding:10px; border-radius:5px;'>"
                  f"Risk Level: {risk_percent}%</div>", unsafe_allow_html=True)
    
    with col2:
        current_price = get_current_price(pair)
        if current_price:
            levels = calculate_levels(pair)
            if levels:
                try:
                    stop_loss_distance = abs(current_price - levels['stop_loss'])
                    position_size = calculate_position_size(account_size, risk_percent, stop_loss_distance)
                    notional_value = position_size * current_price
                    
                    if notional_value > account_size * 2:
                        st.warning("⚠️ Position exceeds 2x account leverage")
                    
                    st.write("## Trading Signals")
                    
                    # Strategy-based signals
                    cols = st.columns(3)
                    cols[0].metric("Buy Zone", f"£{levels['buy_zone']:,.2f}",
                                  delta=f"{levels['buy_zone'] - current_price:+,.2f}")
                    cols[1].metric("Take Profit", f"£{levels['take_profit']:,.2f}",
                                  delta=f"+{(levels['take_profit']/levels['buy_zone']-1)*100:.1f}%")
                    cols[2].metric("Stop Loss", f"£{levels['stop_loss']:,.2f}",
                                  delta=f"-{(1-levels['stop_loss']/levels['buy_zone'])*100:.1f}%", 
                                  delta_color="inverse")
                    
                    st.write("## Risk Management")
                    risk_cols = st.columns(3)
                    risk_cols[0].metric("Current Price", f"£{current_price:,.2f}")
                    risk_cols[1].metric("Position Size", f"{position_size:,.4f} {pair.split('-')[0]}")
                    risk_cols[2].metric("Position Value", f"£{notional_value:,.2f}")

                    # Price indicator chart
                    fig = go.Figure(go.Indicator(
                        mode="number+delta",
                        value=current_price,
                        number={'prefix': "£", 'valueformat': ".2f"},
                        delta={'reference': levels['buy_zone'], 'relative': False},
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Strategy details expander
                    with st.expander("Strategy Parameters"):
                        params = RISK_STRATEGIES[st.session_state.selected_strategy]
                        st.write(f"**Stop Loss:** {params['stop_loss_pct']*100}% from buy zone")
                        st.write(f"**Take Profit:** {params['take_profit_pct']*100}% from buy zone")
                        st.progress(params['take_profit_pct'] / (params['take_profit_pct'] + params['stop_loss_pct']))
                    
                except Exception as e:
                    st.error(f"Display error: {str(e)}")
            else:
                st.error("Insufficient market data for analysis")
        else:
            st.error("Couldn't fetch current prices. Try again later.")

if __name__ == "__main__":
    main()
