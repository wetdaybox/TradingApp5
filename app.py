import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pytz
import requests
import os
import joblib
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import SGDClassifier  # for persistent online learning

# ======================================================
# Configuration & Session Setup
# ======================================================
CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
FX_PAIR = 'GBPUSD=X'
UK_TIMEZONE = pytz.timezone('Europe/London')
REFRESH_INTERVAL = 60  # seconds between auto-refresh

# RSI thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

MODEL_PATH = "sgd_classifier.pkl"  # Path to save persistent ML classifier

if 'manual_price' not in st.session_state:
    st.session_state.manual_price = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")

# ======================================================
# Custom CSS for styling
# ======================================================
custom_css = """
<style>
/* Set a custom font and adjust colors */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f5f5;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf, #2e7bcf);
    color: white;
}
h1 {
    color: #2e7bcf;
    text-align: center;
}
.metric-box {
    background-color: white;
    padding: 10px;
    border-radius: 8px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ======================================================
# Helper Functions
# ======================================================
def format_price(price):
    """Return price as formatted string with appropriate precision.
    Converts input to a float if necessary.
    """
    try:
        if isinstance(price, pd.Series):
            price = price.iloc[0]
        price = float(price)
    except Exception as e:
        st.error(f"format_price conversion error: {e}")
        return str(price)
    
    if price < 10:
        return f"£{price:.8f}"
    elif price < 100:
        return f"£{price:.4f}"
    else:
        return f"£{price:.2f}"

# ======================================================
# Technical Indicator Functions
# ======================================================
def get_rsi(data, window=14):
    if len(data) < window + 1:
        return pd.Series([None] * len(data), index=data.index)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def get_bollinger_bands(data, window=20, num_std=2):
    sma = data['Close'].rolling(window).mean()
    std = data['Close'].rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma, upper, lower

def get_stochastic(data, window=14, smooth_k=3, smooth_d=3):
    low_min = data['Low'].rolling(window).min()
    high_max = data['High'].rolling(window).max()
    k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    k_smooth = k.rolling(smooth_k).mean()
    d = k_smooth.rolling(smooth_d).mean()
    return k_smooth, d

# ======================================================
# Machine Learning Functions (Persistent Classifier)
# ======================================================
def predict_next_return(data, lookback=20):
    """Basic ML forecast using linear regression on log returns."""
    if len(data) < lookback + 1:
        return 0
    data = data.copy()
    data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
    recent = data['LogReturn'].dropna().iloc[-lookback:]
    x = np.arange(len(recent))
    y = recent.values
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0] * 100  # predicted return (%) per period

def ml_classifier_signal(data, lookback=50):
    """
    Uses an online SGDClassifier (with log_loss) as a persistent logistic regression model.
    It extracts features (RSI, MACD, StochK, Return) from the data,
    then updates the model via partial_fit and predicts whether the next period's return is positive.
    Returns: 1 for Buy, -1 for Sell.
    """
    df = data.copy()
    df['Return'] = df['Close'].pct_change()
    df = df.dropna()
    features = df[['RSI', 'MACD', 'StochK', 'Return']].values
    target = (df['Return'].shift(-1) > 0).astype(int).dropna().values
    features = features[:-1]
    if len(features) < lookback:
        lookback = len(features)
    X_train = features[-lookback:]
    y_train = target[-lookback:]
    
    # Load or initialize the persistent model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
    
    try:
        model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
        joblib.dump(model, MODEL_PATH)
        pred = model.predict(X_train[-1].reshape(1, -1))[0]
        return 1 if pred == 1 else -1
    except Exception as e:
        st.error(f"ML classifier persistent error: {e}")
        return 0

# ======================================================
# Sentiment Analysis Function
# ======================================================
def get_sentiment(pair):
    """
    Fetch news headlines for the given pair using yfinance's ticker.news,
    then analyze sentiment using VADER.
    Returns "Positive", "Neutral", or "Negative".
    """
    try:
        ticker = yf.Ticker(pair)
        news = ticker.news
        if not news:
            return "Neutral"
        headlines = " ".join([item.get('title', '') for item in news])
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(headlines)
        compound = score.get("compound", 0)
        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        st.error(f"Sentiment error: {e}")
        return "Neutral"

# ======================================================
# Data Fetching Functions
# ======================================================
@st.cache_data(ttl=30)
def get_realtime_data(pair):
    try:
        data = yf.download(pair, period='2d', interval='5m', progress=False, auto_adjust=True)
        if not data.empty:
            data.index = pd.to_datetime(data.index)
            data['RSI'] = get_rsi(data)
            macd_line, signal_line, _ = get_macd(data)
            data['MACD'] = macd_line
            data['MACD_Signal'] = signal_line
            sma, upper, lower = get_bollinger_bands(data)
            data['SMA'] = sma
            data['UpperBB'] = upper
            data['LowerBB'] = lower
            k, d = get_stochastic(data)
            data['StochK'] = k
            data['StochD'] = d
            st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
        return data
    except Exception as e:
        st.error(f"Data error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_fx_rate():
    try:
        fx_data = yf.download(FX_PAIR, period='1d', interval='5m', progress=False, auto_adjust=True)
        return fx_data['Close'].iloc[-1].item() if not fx_data.empty else 0.80
    except Exception as e:
        st.error(f"FX error: {e}")
        return 0.80

def cross_reference_price(pair):
    try:
        ticker = yf.Ticker(pair)
        alt_data = ticker.history(period='1d', interval='1m', auto_adjust=True)
        if not alt_data.empty:
            return alt_data['Close'].iloc[-1].item()
        else:
            return None
    except Exception as e:
        st.error(f"Alternative data error: {e}")
        return None

def get_price_data(pair):
    # Use yfinance exclusively for consistency
    data = get_realtime_data(pair)
    fx_rate = get_fx_rate()
    if st.session_state.manual_price is not None:
        return st.session_state.manual_price, True
    if not data.empty and 'Close' in data.columns:
        primary_usd = data['Close'].iloc[-1].item()
        return primary_usd / fx_rate, False
    return None, False

# ======================================================
# Signal Aggregator (Ensemble of Indicators)
# ======================================================
def aggregate_signals(data, levels, ml_return, classifier_signal):
    """
    Combine signals from multiple indicators:
      - RSI: Buy if < 30, Sell if > 70.
      - MACD: Buy if MACD > MACD_Signal, Sell if MACD < MACD_Signal.
      - Bollinger Bands: Buy if Close <= LowerBB, Sell if Close >= UpperBB.
      - Stochastic: Buy if StochK < 20, Sell if StochK > 80.
      - ML Forecast: Buy if predicted return > 0.05%, Sell if < -0.05%.
      - ML Classifier: Signal from persistent logistic regression.
    Returns final signal: +1 = Buy, -1 = Sell, 0 = Hold.
    """
    signals = []
    # RSI signal
    rsi_value = levels.get('rsi', 50)
    if rsi_value < RSI_OVERSOLD:
        signals.append(1)
    elif rsi_value > RSI_OVERBOUGHT:
        signals.append(-1)
    else:
        signals.append(0)
    # MACD signal
    try:
        macd = data['MACD'].iloc[-1].item() if not pd.isna(data['MACD'].iloc[-1]) else 0
        macd_signal = data['MACD_Signal'].iloc[-1].item() if not pd.isna(data['MACD_Signal'].iloc[-1]) else 0
        if macd > macd_signal:
            signals.append(1)
        elif macd < macd_signal:
            signals.append(-1)
        else:
            signals.append(0)
    except (IndexError, KeyError):
        signals.append(0)
    # Bollinger Bands signal
    try:
        current_close = data['Close'].iloc[-1].item()
        lower_bb = data['LowerBB'].iloc[-1].item()
        upper_bb = data['UpperBB'].iloc[-1].item()
        if current_close <= lower_bb:
            signals.append(1)
        elif current_close >= upper_bb:
            signals.append(-1)
        else:
            signals.append(0)
    except (IndexError, KeyError):
        signals.append(0)
    # Stochastic signal
    try:
        stoch_k = data['StochK'].iloc[-1].item() if not pd.isna(data['StochK'].iloc[-1]) else 50
        if stoch_k < 20:
            signals.append(1)
        elif stoch_k > 80:
            signals.append(-1)
        else:
            signals.append(0)
    except (IndexError, KeyError):
        signals.append(0)
    # ML forecast signal
    if ml_return > 0.05:
        signals.append(1)
    elif ml_return < -0.05:
        signals.append(-1)
    else:
        signals.append(0)
    # ML classifier signal from persistent model
    signals.append(classifier_signal)
    
    signal_sum = np.sum(signals)
    if signal_sum >= 3:
        return 1
    elif signal_sum <= -3:
        return -1
    else:
        return 0

# ======================================================
# Calculation of Levels and Backtesting
# ======================================================
def calculate_levels(pair, current_price, tp_percent, sl_percent):
    data = get_realtime_data(pair)
    if data.empty:
        return None
    # Filter data for the last 24 hours
    end_time = data.index.max()
    data_24h = data[data.index >= end_time - pd.Timedelta(hours=24)]
    if data_24h.empty:
        return None
    try:
        # Calculate robust low/high in USD then convert to GBP using FX rate
        recent_low = float(data_24h['Low'].quantile(0.05).values[0])
        recent_high = float(data_24h['High'].quantile(0.95).values[0])
        fx_rate = get_fx_rate()
        last_rsi = float(data['RSI'].iloc[-1]) if not pd.isna(data['RSI'].iloc[-1]) else 50
        
        # ATR calculation on 24h data
        high_low = data_24h['High'] - data_24h['Low']
        high_close = (data_24h['High'] - data_24h['Close'].shift()).abs()
        low_close = (data_24h['Low'] - data_24h['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = float(true_range.rolling(14).mean().iloc[-1])
        
        vol = (atr / current_price) * 100  # volatility as % of current price
        volatility = round(vol, 2)
        
        # Convert robust low and high to GBP
        robust_low_local = recent_low / fx_rate
        robust_high_local = recent_high / fx_rate
        
        # Entry zone as the midpoint between current price and robust low (both in GBP)
        buy_zone = round((current_price + robust_low_local) / 2, 2)
        
        return {
            'buy_zone': buy_zone,
            'take_profit': round(current_price * (1 + tp_percent / 100), 2),
            'stop_loss': round(current_price * (1 - sl_percent / 100), 2),
            'rsi': round(last_rsi, 1),
            'high': round(robust_high_local, 2),
            'low': round(robust_low_local, 2),
            'volatility': volatility
        }
    except Exception as e:
        st.error(f"Calculation error: {e}")
        return None

def backtest_strategy(pair, tp_percent, sl_percent, initial_capital=1000):
    data = get_realtime_data(pair)
    if data.empty:
        return None
    fx_rate = get_fx_rate()
    df = data.copy()
    df['Price'] = df['Close'] / fx_rate
    position = 0
    cash = initial_capital
    portfolio = [initial_capital]
    for i in range(1, len(df)):
        current_price = df['Price'].iloc[i].item()
        current_rsi = df['RSI'].iloc[i].item() if not pd.isna(df['RSI'].iloc[i]) else 50
        if position > 0 and current_rsi > RSI_OVERBOUGHT:
            cash = position * current_price
            position = 0
        elif position == 0 and current_rsi < RSI_OVERSOLD:
            position = cash / current_price
            cash = 0
        portfolio.append(cash + position * current_price)
    df = df.iloc[1:].copy()
    df['Portfolio'] = portfolio[1:]
    total_return = ((portfolio[-1] - initial_capital) / initial_capital) * 100
    return df, total_return

# ======================================================
# Main Application
# ======================================================
def main():
    # Header with title and optional logo
    st.image("https://via.placeholder.com/150", width=100)  # Replace URL with your logo
    st.title("🚀 Revolutionary Crypto Trading Bot")
    st.markdown("**Free-to-use, advanced crypto trading assistant**")
    
    # Sidebar for user controls
    st.sidebar.header("Trading Parameters")
    pair = st.sidebar.selectbox("Select Asset:", CRYPTO_PAIRS, help="Choose the crypto asset to trade.")
    use_manual = st.sidebar.checkbox("Enter Price Manually", help="Override the live price with a manual value.")
    if use_manual:
        st.session_state.manual_price = st.sidebar.number_input("Manual Price (£)", min_value=0.01,
                                                                value=st.session_state.manual_price or 1000.0)
    else:
        st.session_state.manual_price = None
    account_size = st.sidebar.number_input("Portfolio Value (£)", min_value=100.0, value=1000.0, step=100.0,
                                             help="Total portfolio value in GBP.")
    risk_profile = st.sidebar.select_slider("Risk Profile:", options=['Safety First', 'Balanced', 'High Risk'],
                                              help="Select your preferred risk profile.")
    risk_reward = st.sidebar.slider("Risk/Reward Ratio", 1.0, 5.0, 3.0, 0.5,
                                    help="Desired risk to reward ratio.")
    tp_percent = st.sidebar.slider("Take Profit %", 1.0, 30.0, 15.0, help="Percentage for profit target.")
    sl_percent = st.sidebar.slider("Stop Loss %", 1.0, 10.0, 5.0, help="Percentage for stop loss.")
    backtest_button = st.sidebar.button("Run Backtest")
    
    # Main area: display update time and metrics
    col1, col2 = st.columns(2)
    with col1:
        update_diff = (datetime.now() - datetime.strptime(st.session_state.last_update, "%H:%M:%S")).seconds
        recency_color = "green" if update_diff < 120 else "orange" if update_diff < 300 else "red"
        st.markdown(f"🕒 Last update: <span style='color:{recency_color}'>{st.session_state.last_update}</span>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown("<p style='text-align: right;'>All values in GBP</p>", unsafe_allow_html=True)
    
    current_price, _ = get_price_data(pair)
    alt_price = cross_reference_price(pair)
    if current_price and alt_price:
        alt_price_gbp = alt_price / get_fx_rate()
        diff_pct = abs(current_price - alt_price_gbp) / current_price * 100
        st.metric("Price Diff (%)", f"{diff_pct:.2f}%", help="Difference between primary and alternative price feeds.")
        st.write(f"**Alternative Price (converted):** {format_price(alt_price_gbp)}")
    
    # ML Forecast Signal
    data = get_realtime_data(pair)
    if not data.empty:
        ml_return = predict_next_return(data, lookback=20)
        ml_signal = "Buy" if ml_return > 0.05 else "Sell" if ml_return < -0.05 else "Hold"
        st.metric("ML Signal", ml_signal, delta=f"{ml_return:.2f}%", help="Forecasted percentage change for next period.")
    
    # Persistent ML Classifier Signal
    if not data.empty:
        classifier_signal = ml_classifier_signal(data, lookback=50)
        st.metric("ML Classifier Signal", "Buy" if classifier_signal == 1 else "Sell" if classifier_signal == -1 else "Hold",
                  help="Classifier prediction based on technical indicators.")
    
    # News Sentiment
    sentiment = get_sentiment(pair)
    st.metric("News Sentiment", sentiment, help="Overall sentiment from recent news headlines.")
    
    # Technical Levels & Ensemble Signal
    if current_price:
        levels = calculate_levels(pair, current_price, tp_percent, sl_percent)
        if levels:
            ensemble_signal = aggregate_signals(data, levels, ml_return, classifier_signal)
            final_signal = "Buy" if ensemble_signal == 1 else "Sell" if ensemble_signal == -1 else "Hold"
            
            st.subheader("Technical Metrics")
            met_cols = st.columns(3)
            rsi_color = "green" if levels['rsi'] < RSI_OVERSOLD else "red" if levels['rsi'] > RSI_OVERBOUGHT else "gray"
            met_cols[0].markdown(f"<span style='color:{rsi_color};font-size:24px'>{levels['rsi']:.1f}</span>",
                                 unsafe_allow_html=True)
            met_cols[0].caption("RSI (Oversold <30, Overbought >70)")
            met_cols[1].metric("24h Range", f"{format_price(levels['low'])} - {format_price(levels['high'])}",
                                help="Robust low/high from the last 24 hours.")
            met_cols[2].metric("Volatility (% of price)", f"{levels['volatility']:.2f}%",
                                help="ATR over 14 periods as a percentage of current price.")
            
            st.markdown(f"**Ensemble Trading Signal: {final_signal}**")
            
            with st.expander("Trading Strategy Details and Explanations"):
                st.write(f"""
                **Recommended Action:** {final_signal}
                
                **Entry Zone:** {format_price(levels['buy_zone'])}  
                **Profit Target:** {format_price(levels['take_profit'])} (+{tp_percent}%)  
                **Stop Loss:** {format_price(levels['stop_loss'])} (-{sl_percent}%)
                """)
                st.markdown("""
                **Explanations:**
                - **RSI:** Indicates market momentum.
                - **24h Range:** Uses the 5th and 95th percentiles of the last 24h prices.
                - **Volatility:** Shows the average true range relative to the current price.
                - **Ensemble Signal:** Combines multiple indicators including ML forecasts.
                """)
                
                # Enhanced interactive chart with range slider
                hist_data = get_realtime_data(pair)
                if hist_data.empty:
                    st.error("Historical data not available for chart display.")
                else:
                    hist_data.index = pd.to_datetime(hist_data.index)
                    hist_data_reset = hist_data.reset_index()
                    date_col = None
                    for col in hist_data_reset.columns:
                        if pd.api.types.is_datetime64_any_dtype(hist_data_reset[col]):
                            date_col = col
                            break
                    if date_col is None:
                        date_col = hist_data_reset.columns[0]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist_data_reset[date_col],
                        y=hist_data_reset["Close"],
                        name="Price History",
                        line=dict(color="#1f77b4")
                    ))
                    fig.add_hline(y=levels["buy_zone"], line_dash="dot",
                                  annotation_text="Buy Zone", line_color="green")
                    fig.add_hline(y=levels["take_profit"], line_dash="dot",
                                  annotation_text="Profit Target", line_color="blue")
                    fig.add_hline(y=levels["stop_loss"], line_dash="dot",
                                  annotation_text="Stop Loss", line_color="red")
                    fig.update_layout(
                        title=f"Historical Price Chart for {pair}",
                        xaxis_title="Date/Time",
                        yaxis_title="Price (£)",
                        template="plotly_white",
                        xaxis=dict(rangeslider=dict(visible=True))
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Position Builder")
            risk_amount = st.slider("Risk Amount (£)", 10.0, account_size, 100.0,
                                    help="Amount of capital at risk per trade.")
            position_size = risk_amount / abs(current_price - levels["stop_loss"])
            st.write(f"""
            **Suggested Position:**  
            - **Size:** {position_size:.6f} {pair.split('-')[0]}  
            - **Value:** {format_price(position_size * current_price)}  
            - **Risk/Reward Ratio:** 1:{risk_reward:.1f}
            """)
        else:
            st.error("Market data unavailable")
    else:
        st.warning("Waiting for price data...")
    
    if backtest_button:
        with st.spinner("Running backtest..."):
            backtest_result = backtest_strategy(pair, tp_percent, sl_percent, initial_capital=account_size)
            if backtest_result is not None:
                bt_data, total_return = backtest_result
                st.subheader("Backtest Results")
                st.line_chart(bt_data["Portfolio"])
                st.write(f"**Strategy Return:** {total_return:.2f}%")
                buy_hold_return = ((bt_data['Price'].iloc[-1] - bt_data['Price'].iloc[0]) / bt_data['Price'].iloc[0]) * 100
                st.write(f"**Buy & Hold Return:** {buy_hold_return:.2f}%")
            else:
                st.error("Backtest failed - insufficient data")
    
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        **Price Diff (%):** Difference between the primary price feed and an alternative data source.
        
        **ML Signal:** Forecasted percentage change using linear regression on log returns.
        
        **ML Classifier Signal:** Prediction based on a persistent logistic regression model updated via online learning.
        
        **News Sentiment:** Overall sentiment derived from recent news headlines.
        
        **RSI:** Momentum indicator (values <30 indicate oversold; >70 indicate overbought).
        
        **24h Range:** Robust low/high prices over the last 24 hours (converted to GBP).
        
        **Volatility:** ATR over 14 periods as a percentage of the current price.
        
        **Ensemble Signal:** Combined indicator consensus.
        
        **Position Builder:** Suggested trade size based on your risk amount and stop loss gap.
        
        **Backtest Results:** Historical simulation of strategy performance.
        """)
    
if __name__ == "__main__":
    main()
