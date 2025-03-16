import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# Configuration - Unchanged
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'

# NEW: Risk management parameters
RISK_STRATEGIES = {
    'Conservative': {'stop_loss_pct': 0.15, 'take_profit_pct': 0.25},
    'Moderate': {'stop_loss_pct': 0.25, 'take_profit_pct': 0.50},
    'Aggressive': {'stop_loss_pct': 0.35, 'take_profit_pct': 0.75}
}

def calculate_position_size(account_size, risk_percent, stop_loss_distance):
    """Enhanced position sizing with validation"""
    try:
        stop_loss_distance = max(0.0001, float(stop_loss_distance))  # Prevent division by zero
        risk_amount = account_size * (risk_percent / 100)
        
        # NEW: Position size limits
        position_size = min(
            risk_amount / stop_loss_distance,
            account_size * 2  # Max 2x account size
        )
        
        return round(position_size, 4)
    except Exception as e:
        st.error(f"Position error: {str(e)}")
        return 0.0

def main():
    st.set_page_config(page_title="Crypto Trader", layout="centered")
    st.title("🇬🇧 Free Crypto Trading Bot")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        pair = st.selectbox("Select Crypto Pair:", CRYPTO_PAIRS)
        account_size = st.number_input("Account Balance (£)", 
                                     min_value=100.0, max_value=1000000.0, 
                                     value=1000.0, step=100.0)
        
        # NEW: Risk strategy selector
        risk_strategy = st.selectbox("Trading Strategy:", list(RISK_STRATEGIES.keys()))
        strategy_params = RISK_STRATEGIES[risk_strategy]
        
        # NEW: Visual risk indicator
        risk_percent = st.slider("Risk Percentage:", 1, 10, 2,
                               help="Percentage of account to risk per trade")
        risk_color = "#FF4B4B" if risk_percent > 5 else "#00CC96"
        st.markdown(f"<div style='background:{risk_color}; padding:10px; border-radius:5px;'>"
                  f"Risk Level: {risk_percent}%</div>", unsafe_allow_html=True)
    
    with col2:
        current_price = get_current_price(pair)  # Keep original pricing logic
        if current_price:
            levels = calculate_levels(pair)  # Keep original level calculation
            if levels:
                try:
                    stop_loss_distance = abs(current_price - levels['stop_loss'])
                    
                    # NEW: Enhanced position sizing
                    position_size = calculate_position_size(account_size, risk_percent, stop_loss_distance)
                    notional_value = position_size * current_price
                    
                    # NEW: Position validation
                    if notional_value > account_size * 2:
                        st.warning("⚠️ Position exceeds 2x account leverage")
                    
                    st.write("## Live Trading Signals")
                    
                    # NEW: Improved metric display
                    cols = st.columns(3)
                    cols[0].metric("Current Price", f"£{current_price:,.2f}", 
                                 delta=f"Strategy: {risk_strategy}", delta_color="off")
                    cols[1].metric("Position Size", f"{position_size:,.4f} {pair.split('-')[0]}",
                                 help="Includes 2x account size limit")
                    cols[2].metric("Risk Amount", f"£{account_size*(risk_percent/100):,.2f}",
                                 delta_color="inverse")
                    
                    # NEW: Strategy parameters display
                    with st.expander("Strategy Details"):
                        st.write(f"**Stop Loss:** {strategy_params['stop_loss_pct']*100}%")
                        st.write(f"**Take Profit:** {strategy_params['take_profit_pct']*100}%")
                        st.progress(strategy_params['stop_loss_pct'] / 
                                   strategy_params['take_profit_pct'])

                    # Keep original chart
                    fig = go.Figure(go.Indicator(
                        mode="number+delta",
                        value=current_price,
                        number={'prefix': "£", 'valueformat': ".2f"},
                        delta={'reference': levels['buy_zone'], 'relative': False},
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Display error: {str(e)}")
            else:
                st.error("Insufficient market data for analysis")
        else:
            st.error("Couldn't fetch current prices. Try again later.")

# REST OF ORIGINAL FUNCTIONS REMAIN UNCHANGED
# get_realtime_data, get_fx_rate, get_current_price, calculate_levels
# ...

if __name__ == "__main__":
    main()
