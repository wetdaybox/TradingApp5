```python
import streamlit as st
import requests
from datetime import datetime

# --- Helper functions ---
@st.cache_data(ttl=60)
def fetch_ticker(instrument: str):
    """Fetch ticker data for a given instrument from Crypto.com Exchange v2 API."""
    url = "https://api.crypto.com/v2/public/get-ticker"
    params = {"instrument_name": instrument}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"API error for {instrument}: {data.get('message', data)}")
    return data["result"]


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
    btc_data = fetch_ticker("BTC_USDT")
    xrp_data = fetch_ticker("XRP_BTC")
    btc_price = float(btc_data["last"])
    # Crypto.com returns 24h change as a string percentage, e.g. "1.23"
    btc_change = float(btc_data.get("change24h", 0))
    xrp_price = float(xrp_data["last"])
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Display prices
col1, col2 = st.columns(2)
with col1:
    st.metric(label="BTC/USDT Price", value=f"${btc_price:,.2f}", delta=f"{btc_change:.2f}%")
with col2:
    st.metric(label="XRP/BTC Price", value=f"{xrp_price:.8f} BTC")

# Trigger logic based on 24h BTC change
trigger = btc_change >= 0.82
if trigger:
    if btc_change <= 4.19:
        drop_pct = 7.22
    else:
        drop_pct = 13.9
    st.markdown(f"## 🔔 **TRIGGER**: BTC has risen {btc_change:.2f}% in the last 24h")
else:
    drop_pct = 0.0
    st.markdown(f"## No trigger (BTC up {btc_change:.2f}% < 0.82%)")

# Sidebar inputs for grid configuration
st.sidebar.header("Grid Configuration")
investment = st.sidebar.number_input(
    "Total investment (in XRP)", min_value=0.0, value=1000.0, step=100.0
)
levels = st.sidebar.number_input(
    "Number of grid levels", min_value=1, value=10, step=1
)

def display_grid():
    if trigger:
        bottom_price, step_size = compute_grid(xrp_price, drop_pct, levels)
        st.write(f"**Grid range:** Top = {xrp_price:.8f} BTC  |  Bottom = {bottom_price:.8f} BTC  (drop {drop_pct}%)")
        st.write(f"**Grid step size:** {step_size:.8f} BTC per level")
    else:
        st.write("No grid adjustment at this time.")

# Show grid
display_grid()

# Auto-refresh every minute
if st.sidebar.checkbox("Auto-refresh every minute", value=True):
    st.experimental_rerun()
```
