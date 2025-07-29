import streamlit as st
import requests

# --- Helper functions ---
@st.cache_data(ttl=60)
def fetch_prices():
    """
    Fetch BTC/USD and XRP/BTC prices and BTC 24h change from CoinGecko's free API.
    """
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin,ripple",
        "vs_currencies": "usd,btc",
        "include_24hr_change": "true"
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    btc_usd = data["bitcoin"]["usd"]
    btc_change = data["bitcoin"]["usd_24h_change"]  # percent
    xrp_btc = data["ripple"]["btc"]
    return btc_usd, btc_change, xrp_btc

def compute_grid(xrp_price: float, pct: float, levels: int):
    """
    Given top price, drop percentage, and grid count, compute bottom and step.
    """
    bottom = xrp_price * (1 - pct / 100)
    step = (xrp_price - bottom) / levels
    return bottom, step

# --- Streamlit App ---
st.set_page_config(page_title="XRP/BTC Grid Bot", layout="centered")
st.title("🟋 XRP/BTC Grid Bot Dashboard")

# Fetch data
try:
    btc_price, btc_change, xrp_price = fetch_prices()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Display metrics
col1, col2 = st.columns(2)
with col1:
    st.metric(label="BTC/USD Price", value=f"${btc_price:,.2f}", delta=f"{btc_change:.2f}%")
with col2:
    st.metric(label="XRP/BTC Price", value=f"{xrp_price:.8f} BTC")

# Trigger logic
trigger = btc_change >= 0.82
if trigger:
    drop_pct = 7.22 if btc_change <= 4.19 else 13.9
    st.markdown(f"## 🔔 **TRIGGER**: BTC has risen {btc_change:.2f}% in the last 24h")
else:
    drop_pct = 0.0
    st.markdown(f"## No trigger (BTC up {btc_change:.2f}% < 0.82%)")

# Sidebar inputs
st.sidebar.header("Grid Configuration")
investment = st.sidebar.number_input(
    "Total investment (in XRP)", min_value=0.0, value=1000.0, step=100.0
)
levels = st.sidebar.number_input(
    "Number of grid levels", min_value=1, value=10, step=1
)

# Display grid if triggered
if trigger:
    bottom_price, step_size = compute_grid(xrp_price, drop_pct, levels)
    st.write(
        f"**Grid range:** Top = {xrp_price:.8f} BTC  |  "
        f"Bottom = {bottom_price:.8f} BTC  (drop {drop_pct}%)"
    )
    st.write(f"**Grid step size:** {step_size:.8f} BTC per level")
else:
    st.write("No grid adjustment at this time.")

# Auto-refresh every minute
if st.sidebar.checkbox("Auto-refresh every minute", value=True):
    st.experimental_rerun()
