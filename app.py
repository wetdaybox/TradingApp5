import streamlit as st
import requests
import time

# --- Helper functions ---
@st.cache_data(ttl=60)
def fetch_ticker(symbol: str):
    """Fetch ticker data for a given symbol from Crypto.com."""
    url = f"https://api.crypto.com/v1/ticker?symbol={symbol}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get('code') != '0':
        raise RuntimeError(f"API error for {symbol}: {data.get('msg')}")
    return data['data']

def compute_grid(xrp_price: float, pct: float, levels: int):
    """Given top price, drop percentage, and grid count, compute bottom and step."""
    bottom = xrp_price * (1 - pct/100)
    step = (xrp_price - bottom) / levels
    return bottom, step

# --- Streamlit UI ---
st.set_page_config(page_title="XRP/BTC Grid Bot", layout="centered")
st.title("🟋 XRP/BTC Grid Bot Dashboard")

# Fetch data
try:
    btc = fetch_ticker("btcusdt")
    xrp = fetch_ticker("xrpbtc")
    btc_price = float(btc["last"])
    btc_change = float(btc["rose"]) * 100
    xrp_price = float(xrp["last"])
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Display prices
col1, col2 = st.columns(2)
with col1:
    st.metric(label="BTC/USDT Price", value=f"${btc_price:,.2f}", delta=f"{btc_change:.2f}%")
with col2:
    st.metric(label="XRP/BTC Price", value=f"{xrp_price:.8f} BTC")

# Trigger logic
trigger = btc_change >= 0.82
if trigger:
    if btc_change <= 4.19:
        drop_pct = 7.22
    else:
        drop_pct = 13.9
    st.markdown(f"## 🔔 **TRIGGER**: BTC is up {btc_change:.2f}%")
else:
    drop_pct = 0.0
    st.markdown(f"## No trigger (BTC up {btc_change:.2f}% < 0.82%)")

# User inputs
st.sidebar.header("Grid Configuration")
investment = st.sidebar.number_input("Total investment (in XRP)", min_value=0.0, value=1000.0, step=100.0)
levels = st.sidebar.number_input("Number of grid levels", min_value=1, value=10, step=1)

# Compute and display grid if triggered
if trigger:
    bottom, step = compute_grid(xrp_price, drop_pct, levels)
    st.write(f"**Grid range:** Top = {xrp_price:.8f} BTC  |  Bottom = {bottom:.8f} BTC  (drop {drop_pct}%)")
    st.write(f"**Grid step size:** {step:.8f} BTC per level")
else:
    st.write("No grid adjustment at this time.")

# Auto-refresh every 5 minutes
st.experimental_rerun()
