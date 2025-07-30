import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# ──── Configuration ────────────────────────────────────────────────────────
CONFIG = {
    "HISTORY_DAYS": 90,
    "VOL_WINDOW": 14,
    "RSI_WINDOW": 14,
    "SMA_SHORT": 5,
    "SMA_LONG": 20,
    "EMA_TREND": 50,
    "RSI_OVERBOUGHT": 75,
    "GRID_XRP": {"PRIMARY": 20, "FEWER": 10, "MORE": 30},
    "API_TIMEOUT": 10,
    "REFRESH_INTERVAL": 60_000  # 60 seconds
}

# ──── Auto‐refresh ─────────────────────────────────────────────────────────
st_autorefresh(interval=CONFIG["REFRESH_INTERVAL"], key="datarefresh")

# ──── Data Fetching Functions ──────────────────────────────────────────────
@st.cache_data(ttl=600)
def fetch_history(days):
    """Fetch historical BTC/USD data and compute indicators"""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days}
    
    try:
        r = requests.get(url, params=params, timeout=CONFIG["API_TIMEOUT"])
        r.raise_for_status()
        prices = r.json()["prices"]
    except Exception as e:
        st.error(f"⚠️ History API Error: {str(e)}")
        return pd.DataFrame()

    df = pd.DataFrame(prices, columns=["ts", "price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date").resample("D").last().dropna()
    
    # Calculate indicators
    df["return"] = df["price"].pct_change() * 100
    df["vol14"] = df["return"].rolling(CONFIG["VOL_WINDOW"]).std()
    df["sma5"] = df["price"].rolling(CONFIG["SMA_SHORT"]).mean()
    df["sma20"] = df["price"].rolling(CONFIG["SMA_LONG"]).mean()
    df["ema50"] = df["price"].ewm(span=CONFIG["EMA_TREND"], adjust=False).mean()
    
    # Calculate RSI
    delta = df["price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(CONFIG["RSI_WINDOW"], min_periods=1).mean()
    avg_loss = loss.rolling(CONFIG["RSI_WINDOW"], min_periods=1).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    return df.dropna()

@st.cache_data(ttl=60)
def fetch_live():
    """Fetch live prices for BTC and XRP"""
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin,ripple",
        "vs_currencies": "usd,btc",
        "include_24hr_change": "true"
    }
    
    try:
        r = requests.get(url, params=params, timeout=CONFIG["API_TIMEOUT"])
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        st.error(f"⚠️ Live Price API Error: {str(e)}")
        return {}
    
    return {
        "BTC/USDT": (j["bitcoin"]["usd"], j["bitcoin"]["usd_24h_change"]),
        "XRP/BTC": (j["ripple"]["btc"], None)
    }

# ──── Helper Functions ─────────────────────────────────────────────────────
def compute_grid(top, drop_pct, levels):
    """Calculate grid parameters for Crypto.com bot"""
    bottom = top * (1 - drop_pct / 100)
    step = (top - bottom) / levels
    return bottom, step

def format_currency(value, pair):
    """Format currency values appropriately for each pair"""
    if pair == "BTC/USDT":
        return f"{value:,.2f}" if value >= 1 else f"{value:.8f}".rstrip('0').rstrip('.')
    else:  # XRP/BTC
        return f"{value:.8f}".rstrip('0').rstrip('.')

# ──── Main Application ─────────────────────────────────────────────────────
def main():
    # ──── Sidebar Settings ─────────────────────────────────────────────────
    st.sidebar.title("💰 Crypto.com Bot Settings")
    inv_btc = st.sidebar.number_input(
        "Total Investment (BTC)",
        min_value=1e-5, 
        value=0.01,
        step=1e-5, 
        format="%.5f",
        help="Total BTC allocated for grid trading"
    )
    min_order = st.sidebar.number_input(
        "Min Order Size (BTC)",
        min_value=1e-6, 
        value=5e-4,
        step=1e-6, 
        format="%.6f",
        help="Minimum order size required by Crypto.com exchange"
    )
    
    # Display status
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **App Status:**  
    ✅ Auto-refresh enabled (60s)  
    🔄 Data updated on load
    """)
    
    # ──── Load Data ────────────────────────────────────────────────────────
    with st.spinner("Loading market data..."):
        hist = fetch_history(CONFIG["HISTORY_DAYS"])
        live = fetch_live()
    
    if hist.empty or not live:
        st.error("Failed to load required data. Please try again later.")
        return
        
    latest = hist.iloc[-1]
    vol14 = latest["vol14"]
    
    # ──── Page Setup ───────────────────────────────────────────────────────
    st.title("⚡ Crypto.com Grid Bot Optimizer")
    st.caption("Optimized for XRP/BTC and BTC/USDT trading pairs")
    
    # London time display
    now_london = datetime.now(pytz.timezone("Europe/London"))
    st.markdown(f"**Last Updated:** {now_london.strftime('%Y-%m-%d %H:%M %Z')}")
    
    # Information note
    st.info(
        "💡 Grid settings are optimized for Crypto.com's grid bot interface. "
        "Use the values in the code blocks to configure your bot.",
        icon="ℹ️"
    )
    
    # ──── Strategy Backtest ────────────────────────────────────────────────
    mod_th = vol14
    str_th = 2 * vol14
    
    # Backtest conditions
    cond = (
        (hist["return"] >= mod_th) &
        (hist["price"] > hist["ema50"]) &
        (hist["sma5"] > hist["sma20"]) &
        (hist["rsi"] < CONFIG["RSI_OVERBOUGHT"])
    )
    
    # Calculate backtest results
    trades = int(cond.sum())
    wins = int(((hist["price"].shift(-1) > hist["price"]) & cond).sum())
    win_rate = wins / trades if trades else 0
    
    # Optimize grid count for BTC/USDT
    btc_change = live["BTC/USDT"][1]
    drop_pct_btc = mod_th if btc_change < mod_th else (
        str_th if btc_change > str_th else btc_change
    )
    scores = [win_rate * (drop_pct_btc / L) for L in range(1, 31)]
    opt_L = int(np.argmax(scores)) + 1
    few_L = max(1, opt_L - 10)
    mor_L = min(30, opt_L + 10)
    
    # ──── Bot Execution Function ───────────────────────────────────────────
    def run_bot(name, pair, price, pct_change):
        """Execute trading bot logic for a Crypto.com pair"""
        st.subheader(f"🤖 {name} Bot")
        
        # Price and metrics display
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Current Price", format_currency(price, pair))
            if pct_change is not None:
                st.metric("24h Change", f"{pct_change:.2f}%")
            st.metric("14D Volatility", f"{vol14:.2f}%")
        
        # Strategy filters status
        filters_ok = (
            (latest["price"] > latest["ema50"]) and
            (latest["sma5"] > latest["sma20"]) and
            (latest["rsi"] < CONFIG["RSI_OVERBOUGHT"])
        )
        
        with col2:
            st.write("**Strategy Conditions:**")
            st.markdown(f"- **Trend:** `{'✅' if latest['price'] > latest['ema50'] else '❌'}` Price > EMA50")
            st.markdown(f"- **Momentum:** `{'✅' if latest['sma5'] > latest['sma20'] else '❌'}` SMA5 > SMA20")
            st.markdown(f"- **RSI:** `{'✅' if latest['rsi'] < CONFIG['RSI_OVERBOUGHT'] else '❌'}` < {CONFIG['RSI_OVERBOUGHT']}")
            st.markdown(f"- **All Conditions Met:** `{'✅' if filters_ok else '❌'}`")
        
        # Reset status determination
        change = pct_change if pct_change is not None else hist["return"].iloc[-1]
        if change < mod_th:
            drop, status, color = None, f"No reset needed ({change:.2f}% < {mod_th:.2f}%)", "green"
        elif change <= str_th:
            drop, status, color = mod_th, f"Moderate reset (drop {mod_th:.2f}%)", "orange"
        else:
            drop, status, color = str_th, f"Strong reset (drop {str_th:.2f}%)", "red"
        
        st.markdown(f"**Grid Status:** :{color}[{status}]")
        
        # Grid recommendations
        if drop is not None and filters_ok:
            st.success("**✅ GRID ADJUSTMENT RECOMMENDED**")
            
            # Determine grid levels based on pair
            if pair == "BTC/USDT":
                primary, fewer, more = opt_L, few_L, mor_L
                st.info(f"Optimized levels: {opt_L} (Backtest win rate: {win_rate*100:.1f}%)")
            else:
                primary = CONFIG["GRID_XRP"]["PRIMARY"]
                fewer = CONFIG["GRID_XRP"]["FEWER"]
                more = CONFIG["GRID_XRP"]["MORE"]
                st.info(f"Standard XRP/BTC levels: {primary} (Fixed configuration)")
            
            # Calculate grid parameters
            bottom_primary, step_primary = compute_grid(price, drop, primary)
            bottom_fewer, step_fewer = compute_grid(price, drop, fewer)
            bottom_more, step_more = compute_grid(price, drop, more)
            
            # Display Crypto.com ready settings
            st.markdown("### ⚙️ Crypto.com Grid Parameters")
            
            # Most profitable settings
            st.markdown("#### 🥇 Most Profitable Configuration")
            st.code(f"""
Pair: {pair}
Upper Price: {format_currency(price, pair)}
Lower Price: {format_currency(bottom_primary, pair)}
Number of Grids: {primary}
Investment: {inv_btc:.6f} BTC
            """.strip(), language="text")
            
            # Alternative settings
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"#### ⚖️ Fewer Grids ({fewer})")
                st.code(f"""
Upper: {format_currency(price, pair)}
Lower: {format_currency(bottom_fewer, pair)}
Grids: {fewer}
                """.strip(), language="text")
            
            with cols[1]:
                st.markdown(f"#### 🔍 More Grids ({more})")
                st.code(f"""
Upper: {format_currency(price, pair)}
Lower: {format_currency(bottom_more, pair)}
Grids: {more}
                """.strip(), language="text")
            
            # Validation information
            st.markdown("### 🔍 Order Validation")
            per_order = inv_btc / primary
            st.markdown(f"- **Per-order amount:** `{per_order:.6f}` BTC")
            st.markdown(f"- **Minimum required:** `{min_order:.6f}` BTC")
            st.markdown(f"- **Validation:** {'✅' if per_order >= min_order else '❌'} "
                        f"{' (Meets requirements)' if per_order >= min_order else ' (Below minimum)'}")
        else:
            st.warning("**⚠️ MAINTAIN CURRENT SETTINGS**")
            st.info("No grid adjustment recommended at this time based on market conditions.")
        
        st.markdown("---")
    
    # ──── Execute Bots ─────────────────────────────────────────────────────
    run_bot("XRP/BTC", "XRP/BTC", live["XRP/BTC"][0], live["XRP/BTC"][1])
    run_bot("BTC/USDT", "BTC/USDT", live["BTC/USDT"][0], live["BTC/USDT"][1])
    
    # ──── Historical Signals ───────────────────────────────────────────────
    st.subheader("📜 Signal History")
    
    # Create signals dataframe
    signals_df = hist.copy()
    signals_df["Signal"] = cond
    signals_df = signals_df.reset_index()
    signals_df["date"] = signals_df["date"].dt.date
    
    # Filter and format - CRITICAL FIX: Only show actual historical data
    display_df = signals_df[["date", "price", "return", "vol14", "rsi", "Signal"]]
    display_df = display_df.rename(columns={
        "date": "Date",
        "price": "Price",
        "return": "Return %",
        "vol14": "Volatility",
        "rsi": "RSI",
    })
    
    # Format numeric columns
    display_df["Price"] = display_df["Price"].apply(lambda x: f"{x:,.2f}")
    display_df["Return %"] = display_df["Return %"].apply(lambda x: f"{x:.2f}%")
    display_df["Volatility"] = display_df["Volatility"].apply(lambda x: f"{x:.2f}%")
    display_df["RSI"] = display_df["RSI"].apply(lambda x: f"{x:.2f}")
    display_df["Signal"] = display_df["Signal"].apply(lambda x: "✅" if x else "❌")
    
    # Display as table - Show only actual data points
    st.dataframe(display_df, height=400)
    st.caption("BTC/USD daily closing prices from CoinGecko (UTC time)")
    
    # ──── Strategy Analytics ───────────────────────────────────────────────
    st.subheader("📈 Performance Metrics")
    cols = st.columns(3)
    cols[0].metric("Total Signals", trades)
    cols[1].metric("Win Rate", f"{win_rate*100:.1f}%" if trades > 0 else "N/A")
    cols[2].metric("Current Volatility", f"{vol14:.2f}%")
    
    # Signal frequency
    if trades > 0:
        signals_per_day = trades / CONFIG["HISTORY_DAYS"]
        freq_data = {
            "Per Day": signals_per_day,
            "Per Week": signals_per_day * 7,
            "Per Month": signals_per_day * 30,
        }
        
        st.markdown("**Expected Signal Frequency:**")
        for period, value in freq_data.items():
            st.markdown(f"- **{period}:** {value:.2f} signals")

# ──── Run Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
