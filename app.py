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
from sklearn.linear_model import SGDClassifier  # For persistent online learning

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
# Custom CSS for styling (iOS-inspired minimal design)
# ======================================================
custom_css = """
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #f5f5f5;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf, #2e7bcf);
    color: white;
}
h1, h2, h3 {
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
    """Return price as formatted string with appropriate precision."""
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
    """Simple linear regression on log returns for short-term forecast."""
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
# Sentiment Analysis Function (with Fallback & Funny)
# ======================================================
def get_sentiment(pair):
    """
    Fetch news headlines for the given pair using yfinance's ticker.news.
    If no news is returned, use a fallback Yahoo Finance search API endpoint.
    Then, analyze sentiment using VADER by computing individual sentiment scores
    for each headline and averaging them. Returns "Positive", "Neutral", or "Negative".
    
    Tweaked thresholds for sensitivity:
    - Average compound score > 0.1 -> "Positive"
    - Average compound score < -0.1 -> "Negative"
    - Otherwise -> "Neutral"
    """
    try:
        ticker = yf.Ticker(pair)
        news = ticker.news
        if not news:
            # Fallback: use Yahoo Finance search API to fetch news headlines
            search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={pair}&newsCount=5"
            response = requests.get(search_url)
            if response.status_code == 200:
                data = response.json()
                news = data.get("news", [])
        
        headlines = [item.get('title', '').strip() for item in news if item.get('title', '').strip()]
        if not headlines:
            st.info("No meaningful news headlines found. Staying Neutral!")
            return "Neutral"
        
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(title)['compound'] for title in headlines]
        avg_score = np.mean(scores)
        
        st.write("**Debug - Fetched Headlines:**", headlines)
        
        if avg_score > 0.1:
            sentiment = "Positive"
        elif avg_score < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        if sentiment == "Neutral":
            st.info("The news is having an identity crisis—Neutral it is!")
        elif sentiment == "Positive":
            st.success("The news is singing a happy tune: Positive vibes!")
        else:
            st.error("The news is as gloomy as a rainy Monday: Negative vibes!")
        return sentiment
    except Exception as e:
        st.error(f"Sentiment error: {e}. Even the news is confused, so let's stay Neutral.")
        return "Neutral"

# ======================================================
# Data Fetching Functions
# ======================================================
@st.cache_data(ttl=30)
def get_realtime_data(pair):
    """Get 7 days of 5-min data for resampling."""
    try:
        data = yf.download(pair, period='7d', interval='5m', progress=False, auto_adjust=True)
        if not data.empty:
            if 'Adj Close' in data.columns and 'Close' not in data.columns:
                data.rename(columns={'Adj Close': 'Close'}, inplace=True)
            data.index = pd.to_datetime(data.index)
            if 'Close' not in data.columns:
                st.warning(f"No 'Close' column found for {pair} data.")
                return pd.DataFrame()
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
        if 'Adj Close' in fx_data.columns and 'Close' not in fx_data.columns:
            fx_data.rename(columns={'Adj Close': 'Close'}, inplace=True)
        return fx_data['Close'].iloc[-1].item() if not fx_data.empty else 0.80
    except Exception as e:
        st.error(f"FX error: {e}")
        return 0.80

def cross_reference_price(pair):
    try:
        ticker = yf.Ticker(pair)
        alt_data = ticker.history(period='1d', interval='1m', auto_adjust=True)
        if 'Adj Close' in alt_data.columns and 'Close' not in alt_data.columns:
            alt_data.rename(columns={'Adj Close': 'Close'}, inplace=True)
        if not alt_data.empty:
            return alt_data['Close'].iloc[-1].item()
        else:
            return None
    except Exception as e:
        st.error(f"Alternative data error: {e}")
        return None

def get_price_data(pair):
    data = get_realtime_data(pair)
    fx_rate = get_fx_rate()
    if st.session_state.manual_price is not None:
        return st.session_state.manual_price, True
    if not data.empty and 'Close' in data.columns:
        primary_usd = data['Close'].iloc[-1].item()
        return primary_usd / fx_rate, False
    return None, False

# ======================================================
# Weighted Signal Aggregator
# ======================================================
def weighted_aggregate_signals(data, levels, ml_return, classifier_signal):
    weights = {
        'rsi': 1.0,
        'macd': 1.0,
        'bb': 0.8,
        'stoch': 0.8,
        'ml_return': 1.5,
        'classifier': 1.5
    }
    
    signals = []
    rsi_value = levels.get('rsi', 50)
    if rsi_value < RSI_OVERSOLD:
        signals.append(weights['rsi'])
    elif rsi_value > RSI_OVERBOUGHT:
        signals.append(-weights['rsi'])
    else:
        signals.append(0)
    
    try:
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_Signal'].iloc[-1]
        if macd > macd_signal:
            signals.append(weights['macd'])
        elif macd < macd_signal:
            signals.append(-weights['macd'])
        else:
            signals.append(0)
    except Exception:
        signals.append(0)
    
    try:
        current_close = data['Close'].iloc[-1]
        lower_bb = data['LowerBB'].iloc[-1]
        upper_bb = data['UpperBB'].iloc[-1]
        if current_close <= lower_bb:
            signals.append(weights['bb'])
        elif current_close >= upper_bb:
            signals.append(-weights['bb'])
        else:
            signals.append(0)
    except Exception:
        signals.append(0)
    
    try:
        stoch_k = data['StochK'].iloc[-1]
        if stoch_k < 20:
            signals.append(weights['stoch'])
        elif stoch_k > 80:
            signals.append(-weights['stoch'])
        else:
            signals.append(0)
    except Exception:
        signals.append(0)
    
    if ml_return > 0.05:
        signals.append(weights['ml_return'])
    elif ml_return < -0.05:
        signals.append(-weights['ml_return'])
    else:
        signals.append(0)
    
    if classifier_signal == 1:
        signals.append(weights['classifier'])
    elif classifier_signal == -1:
        signals.append(-weights['classifier'])
    else:
        signals.append(0)
    
    total_score = sum(signals)
    decision_threshold = 2.5
    if total_score >= decision_threshold:
        return 1  # Buy
    elif total_score <= -decision_threshold:
        return -1  # Sell
    else:
        return 0  # Hold

# ======================================================
# Levels Calculation and Backtesting with Dynamic ATR & Trend Filter
# ======================================================
def calculate_levels(pair, current_price, tp_multiplier, sl_multiplier, atr_lookback):
    data = get_realtime_data(pair)
    if data.empty:
        return None
    end_time = data.index.max()
    data_24h = data[data.index >= end_time - pd.Timedelta(hours=24)]
    if data_24h.empty:
        return None
    try:
        # Calculate ATR with a variable lookback period
        high_low = data_24h['High'] - data_24h['Low']
        high_close = (data_24h['High'] - data_24h['Close'].shift()).abs()
        low_close = (data_24h['Low'] - data_24h['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(atr_lookback).mean().iloc[-1]
        
        # Dynamic levels using ATR multipliers
        dynamic_tp = round(current_price + (atr * tp_multiplier), 2)
        dynamic_sl = round(current_price - (atr * sl_multiplier), 2)
        
        # Additional metrics
        recent_low = float(data_24h['Low'].quantile(0.05).values[0])
        recent_high = float(data_24h['High'].quantile(0.95).values[0])
        fx_rate = get_fx_rate()
        last_rsi = float(data['RSI'].iloc[-1]) if not pd.isna(data['RSI'].iloc[-1]) else 50
        robust_low_local = recent_low / fx_rate
        robust_high_local = recent_high / fx_rate
        buy_zone = round((current_price + robust_low_local) / 2, 2)
        
        return {
            'buy_zone': buy_zone,
            'take_profit': dynamic_tp,
            'stop_loss': dynamic_sl,
            'rsi': round(last_rsi, 1),
            'high': round(robust_high_local, 2),
            'low': round(robust_low_local, 2),
            'volatility': round((atr / current_price) * 100, 2)
        }
    except Exception as e:
        st.error(f"Calculation error: {e}")
        return None

def backtest_strategy(pair, tp_multiplier, sl_multiplier, atr_lookback, trailing_stop_percent, initial_capital=1000):
    data = get_realtime_data(pair)
    if data.empty:
        return None
    fx_rate = get_fx_rate()
    df = data.copy()
    if 'Close' not in df.columns:
        st.error("No 'Close' column found in data for backtest.")
        return None
    df['Price'] = df['Close'] / fx_rate
    
    position = 0
    cash = initial_capital
    portfolio = [initial_capital]
    entry_price = None
    max_price = 0  # for trailing stop
    
    # Calculate initial dynamic levels
    levels = calculate_levels(pair, df['Price'].iloc[0], tp_multiplier, sl_multiplier, atr_lookback)
    if levels is None:
        return None
    fixed_sl = levels['stop_loss']
    
    for i in range(1, len(df)):
        current_price = df['Price'].iloc[i]
        
        # If in a trade, update max price and trailing stop
        if position > 0:
            max_price = max(max_price, current_price)
            trailing_stop = max_price * (1 - trailing_stop_percent / 100)
        else:
            trailing_stop = None
        
        # Exit conditions: price drops below fixed SL or trailing stop (if set)
        if position > 0 and (current_price <= fixed_sl or (trailing_stop and current_price <= trailing_stop)):
            cash = position * current_price
            position = 0
            entry_price = None
        
        # Entry condition: if not in position and price above buy zone
        elif position == 0:
            # Recalculate dynamic levels at entry
            levels = calculate_levels(pair, current_price, tp_multiplier, sl_multiplier, atr_lookback)
            if levels and current_price >= levels['buy_zone']:
                position = cash / current_price
                entry_price = current_price
                max_price = current_price
                fixed_sl = levels['stop_loss']
                cash = 0
        
        portfolio.append(cash + position * current_price)
    
    df = df.iloc[1:].copy()
    df['Portfolio'] = portfolio[1:]
    total_return = ((portfolio[-1] - initial_capital) / initial_capital) * 100
    return df, total_return

# ======================================================
# Main Application with Trend Filter
# ======================================================
def main():
    st.title("🚀 Revolutionary Crypto Trading Bot")
    st.markdown("**Free-to-use, advanced crypto trading assistant**")
    
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
    # ATR multipliers and lookback
    tp_multiplier = st.sidebar.slider("ATR TP Multiplier", 1.0, 5.0, 4.0, 0.5, help="Multiplier for ATR to set take profit level.")
    sl_multiplier = st.sidebar.slider("ATR SL Multiplier", 0.5, 3.0, 1.5, 0.1, help="Multiplier for ATR to set stop loss level.")
    atr_lookback = st.sidebar.slider("ATR Lookback Period", 10, 30, 14, 1, help="Number of periods for ATR calculation.")
    trailing_stop_percent = st.sidebar.slider("Trailing Stop (%)", 0.5, 5.0, 2.0, 0.1, help="Trailing stop percentage to lock in profits.")
    backtest_button = st.sidebar.button("Run Backtest")
    
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
    
    data = get_realtime_data(pair)
    if not data.empty and current_price:
        ml_return = predict_next_return(data, lookback=20)
        ml_signal = "Buy" if ml_return > 0.05 else "Sell" if ml_return < -0.05 else "Hold"
        st.metric("ML Signal", ml_signal, delta=f"{ml_return:.2f}%", help="Forecasted % change for next period.")
        
        classifier_signal = ml_classifier_signal(data, lookback=50)
        st.metric("ML Classifier Signal", "Buy" if classifier_signal == 1 else "Sell" if classifier_signal == -1 else "Hold",
                  help="Classifier prediction based on technical indicators.")
        
        sentiment = get_sentiment(pair)
        st.metric("News Sentiment", sentiment, help="Overall sentiment from recent news headlines.")
        
        # Calculate dynamic levels
        levels = calculate_levels(pair, current_price, tp_multiplier, sl_multiplier, atr_lookback)
        
        # Apply Trend Filter using EMA(50) on real-time data
        if not data.empty and 'Close' in data.columns:
            ema50 = data['Close'].ewm(span=50, adjust=False).mean().iloc[-1] / get_fx_rate()
            trend = "Bullish" if current_price >= ema50 else "Bearish"
            st.write(f"**Trend Filter (EMA50): {trend}**")
        else:
            trend = "Neutral"
        
        # Get ensemble signal
        if levels:
            ensemble_signal = weighted_aggregate_signals(data, levels, ml_return, classifier_signal)
            # Override signal if it conflicts with the trend
            if trend == "Bullish" and ensemble_signal == -1:
                ensemble_signal = 0
            elif trend == "Bearish" and ensemble_signal == 1:
                ensemble_signal = 0
            final_signal = "Buy" if ensemble_signal == 1 else "Sell" if ensemble_signal == -1 else "Hold"
            
            st.subheader("Technical Metrics")
            tech_cols = st.columns(4)
            tech_cols[0].metric("RSI", f"{levels['rsi']:.1f}", help="Oversold if <30, Overbought if >70")
            tech_cols[1].metric("24h Low", format_price(levels['low']), help="Robust low of last 24h")
            tech_cols[2].metric("24h High", format_price(levels['high']), help="Robust high of last 24h")
            tech_cols[3].metric("Volatility (% of price)", f"{levels['volatility']:.2f}%", help="ATR based volatility")
            
            st.markdown(f"**Ensemble Trading Signal: {final_signal}**")
            
            with st.expander("Trading Strategy Details and Explanations"):
                st.write(f"""
                **Recommended Action:** {final_signal}
                
                **Entry Zone:** {format_price(levels['buy_zone'])}  
                **Take Profit (ATR-Based):** {format_price(levels['take_profit'])}  
                **Stop Loss (ATR-Based):** {format_price(levels['stop_loss'])}
                """)
                st.markdown("""
                **Explanations:**
                - **RSI:** Momentum indicator (<30 oversold, >70 overbought).
                - **24h Low/High:** 5th/95th percentile over the last 24h in GBP.
                - **ATR-Based Levels:** Dynamic take profit and stop loss levels based on recent volatility.
                - **Trend Filter:** EMA(50) to ensure trade direction aligns with overall trend.
                - **Ensemble Signal:** Weighted combination of multiple indicators and ML predictions.
                """)
                
                daily_data = data.copy()
                if 'Adj Close' in daily_data.columns and 'Close' not in daily_data.columns:
                    daily_data.rename(columns={'Adj Close': 'Close'}, inplace=True)
                if 'Close' in daily_data.columns:
                    daily_data = daily_data[['Close']].dropna()
                    daily_data = daily_data.resample("1D").last().dropna()
                    daily_data.index.name = "Date"
                    daily_data = daily_data.reset_index()
                    
                    if daily_data.empty:
                        st.warning("No daily data available for chart display.")
                    else:
                        fig = go.Figure(go.Bar(
                            x=daily_data["Date"],
                            y=daily_data["Close"],
                            marker_color="#2e7bcf",
                            name="Daily Price"
                        ))
                        fig.add_hline(y=levels["buy_zone"], line_dash="dot",
                                      annotation_text="Buy Zone", line_color="green", annotation_position="bottom left")
                        fig.add_hline(y=levels["take_profit"], line_dash="dot",
                                      annotation_text="Take Profit", line_color="blue", annotation_position="top left")
                        fig.add_hline(y=levels["stop_loss"], line_dash="dot",
                                      annotation_text="Stop Loss", line_color="red", annotation_position="top right")
                        fig.update_layout(
                            title=dict(text=f"Daily Bar Chart for {pair}", x=0.5, font=dict(size=18)),
                            xaxis=dict(
                                title="Date",
                                showgrid=True,
                                gridcolor="#e1e1e1"
                            ),
                            yaxis=dict(
                                title="Price (£)",
                                showgrid=True,
                                gridcolor="#e1e1e1"
                            ),
                            paper_bgcolor="white",
                            plot_bgcolor="white",
                            font=dict(
                                family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
                                color="#333"
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No 'Close' data available for daily chart.")
            
            st.subheader("Position Builder")
            risk_amount = st.slider("Risk Amount (£)", 10.0, account_size, 100.0,
                                    help="Amount of capital at risk per trade.")
            position_size = risk_amount / abs(current_price - levels["stop_loss"])
            st.write(f"""
            **Suggested Position:**  
            - **Size:** {position_size:.6f} {pair.split('-')[0]}  
            - **Value:** {format_price(position_size * current_price)}  
            - **Trailing Stop:** {trailing_stop_percent}% for dynamic exit management
            """)
        else:
            st.error("Market data unavailable for strategy levels.")
    else:
        st.warning("Waiting for price data...")
    
    if backtest_button:
        with st.spinner("Running backtest..."):
            backtest_result = backtest_strategy(pair, tp_multiplier, sl_multiplier, atr_lookback, trailing_stop_percent, initial_capital=account_size)
            if backtest_result is not None:
                bt_data, total_return = backtest_result
                st.subheader("Backtest Results")
                st.bar_chart(bt_data["Portfolio"])
                st.write(f"**Strategy Return:** {total_return:.2f}%")
                buy_hold_return = ((bt_data['Price'].iloc[-1] - bt_data['Price'].iloc[0]) / bt_data['Price'].iloc[0]) * 100
                st.write(f"**Buy & Hold Return:** {buy_hold_return:.2f}%")
            else:
                st.error("Backtest failed - insufficient data")
    
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        **Price Diff (%):** Difference between the primary and alternative price feeds.
        
        **ML Signal:** Forecasted % change (via linear regression on log returns) for the next period.
        
        **ML Classifier Signal:** Prediction from a persistent logistic regression classifier updated via online learning.
        
        **News Sentiment:** Overall sentiment derived from recent news headlines.
        
        **RSI:** Momentum indicator (values below 30 indicate oversold; above 70 indicate overbought).
        
        **24h Low/High:** 5th/95th percentile prices over the last 24 hours in GBP.
        
        **ATR-Based Levels:** Dynamic take profit and stop loss levels based on recent volatility.
        
        **Trend Filter:** EMA(50) used to confirm the prevailing market trend.
        
        **Ensemble Signal:** Weighted combination of multiple indicators and ML predictions.
        
        **Position Builder:** Suggested trade size based on your risk amount and the dynamic stop loss gap.
        
        **Trailing Stop:** An exit mechanism that locks in profits as the trade moves favorably.
        
        **Backtest Results:** Historical simulation of strategy performance.
        """)
    
if __name__ == "__main__":
    main()
