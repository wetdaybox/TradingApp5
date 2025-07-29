import streamlit as st
import requests
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 seconds
st_autorefresh(interval=60_000, key="datarefresh")

# Fixed thresholds
THRESHOLD1 = 0.82
THRESHOLD2 = 4.19
DROP1 = 7.22
DROP2 = 13.9

# --- Fetch live prices from CoinGecko ---
@st.cache_data(ttl=60)
def fetch_prices():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin,ripple",
        "vs_currencies": "usd,btc",
        "include_24hr_change": "true"
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    d = r.json()
    btc_usd = d["bitcoin"]["usd"]
    btc_change = d["bitcoin"]["usd_24h_change"]  # percent
    xrp_btc = d["ripple"]["btc"]
    return btc_usd, btc_change, xrp_btc

# --- Compute grid bounds ---
def compute_grid(top_price, drop_pct, levels):
    bottom = top_price * (1 - drop_pct/100)
    step = (top_price - bottom) / levels
    return bottom, step

# --- Streamlit UI ---
st.title("🟋 XRP/BTC Grid Bot (Simple, Stable)")

# Fetch data
try:
    btc_price, btc_change, xrp_price = fetch_prices()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Show prices
col1, col2 = st.columns(2)
col1.metric("BTC/USD", f"${btc_price:,.2f}", f"{btc_change:.2f}%")
col2.metric("XRP/BTC", f"{xrp_price:.8f} BTC")

# Determine drop_pct
if btc_change < THRESHOLD1:
    drop_pct = None
    st.markdown(f"## No reset (BTC up {btc_change:.2f}% < {THRESHOLD1}%)")
elif btc_change <= THRESHOLD2:
    drop_pct = DROP1
    st.markdown(f"## 🔔 Moderate reset: BTC up {btc_change:.2f}% → drop {DROP1}%")
else:
    drop_pct = DROP2
    st.markdown(f"## 🔔 Strong reset: BTC up {btc_change:.2f}% → drop {DROP2}%")

# Sidebar for grid levels
levels = st.sidebar.number_input("Number of grid levels", min_value=1, value=10, step=1)

# Display grid if triggered
if drop_pct is not None:
    bottom, step = compute_grid(xrp_price, drop_pct, levels)
    st.write(f"**Grid top:** {xrp_price:.8f} BTC | **bottom:** {bottom:.8f} BTC")
    st.write(f"**Step size:** {step:.8f} BTC per level over {levels} levels")
else:
    st.write("No grid adjustment at this time.")
