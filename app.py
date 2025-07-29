import streamlit as st
import requests
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 000 ms (60 s)
st_autorefresh(interval=60_000, key="datarefresh")

# --- Helper functions ---
@st.cache_data(ttl=60)
def fetch_data():
    """
    Fetch BTC/USD price and 24h change, and XRP/BTC price from CoinGecko's free API.
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

def compute_grid(top_price: float, drop_pct: float, levels: int):
    """Compute bottom price and per-level step for the grid."""
    bottom = top_price * (1 - drop_pct / 100)
    step = (top_price - bottom) / levels
    return bottom, step

# --- Streamlit App ---
st.set_page_config(page_title="XRP/BTC Grid Bot", layout="centered")
st.title("🟋 XRP/BTC Grid Bot Dashboard")

# Fetch live data
try:
    btc_price, btc_change, xrp_price = fetch_data()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Display metrics
col1, col2 = st.columns(2)
with col1:
    st.metric("BTC/USD Price", f"${btc_price:,.2f}", f"{btc_change:.2f}%")
with col2:
    st.metric("XRP/BTC Price", f"{xrp_price:.8f} BTC")

# Determine drop percentage based on BTC change
if btc_change < 0.82:
    drop_pct = None  # no reset
elif btc_change <= 4.19:
    drop_pct = 7.22
else:
    drop_pct = 13.9

# Show trigger status
if drop_pct is None:
    st.markdown(f"## No reset (BTC up {btc_change:.2f}% < 0.82%)")
else:
    st.markdown(f"## 🔔 Reset triggered: BTC up {btc_change:.2f}%")

# Sidebar: grid config
st.sidebar.header("Grid Configuration")
levels = st.sidebar.number_input(
    "Number of grid levels", min_value=1, value=10, step=1
)

# Calculate and display grid if triggered
if drop_pct is not None:
    top_price = xrp_price
    bottom_price, step_size = compute_grid(top_price, drop_pct, levels)
    st.write(
        f"**Grid range:** Top = {top_price:.8f} BTC  |  "
        f"Bottom = {bottom_price:.8f} BTC  (drop {drop_pct}%)"
    )
    st.write(f"**Step size:** {step_size:.8f} BTC per level")
else:
    st.write("No grid adjustment at this time.")
