#!/usr/bin/env python3
"""
Backtrader Trading Bot Example

This program uses Backtrader to implement a simple SMA crossover strategy on AAPL data.
It fetches data from Yahoo Finance using yfinance, creates a Backtrader data feed,
runs the strategy, and displays the results using matplotlib.
This is a proven open-source system that many traders have used successfully.
"""

import yfinance as yf
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta

# Define a simple SMA Crossover Strategy using Backtrader
class SmaCross(bt.Strategy):
    params = (('sma_period', 50),)

    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_period)

    def next(self):
        # If not in the market, and close > SMA then buy.
        if not self.position:
            if self.data.close[0] > self.sma[0]:
                self.buy()
        # If in the market, and close < SMA then sell.
        else:
            if self.data.close[0] < self.sma[0]:
                self.sell()

# Function to run Backtrader simulation
def run_backtrader():
    # Define the date range (last 3 years)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*3)
    
    # Fetch historical AAPL data from Yahoo Finance using yfinance
    data = yf.download('AAPL', start=start_date.strftime('%Y-%m-%d'),
                       end=end_date.strftime('%Y-%m-%d'))
    if data.empty:
        st.error("No data fetched from Yahoo Finance.")
        return None

    # Create a Backtrader data feed from the Pandas DataFrame
    datafeed = bt.feeds.PandasData(dataname=data)
    
    # Initialize Cerebro engine and add strategy
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross, sma_period=50)
    cerebro.adddata(datafeed)
    
    # Set initial cash, stake size, and commission
    cerebro.broker.setcash(10000.0)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    cerebro.broker.setcommission(commission=0.001)
    
    st.write("Starting Portfolio Value: ${:.2f}".format(cerebro.broker.getvalue()))
    
    # Run the backtest
    cerebro.run()
    
    final_value = cerebro.broker.getvalue()
    st.write("Final Portfolio Value: ${:.2f}".format(final_value))
    
    # Generate and return the plot (Backtrader uses matplotlib)
    figs = cerebro.plot(style='candlestick')
    # Backtrader.plot() returns nested lists; extract the first figure
    fig = figs[0][0]
    return fig

# Streamlit Interface
st.title("Backtrader Trading Bot Example")
st.markdown("""
This example uses the Backtrader framework to run a simple SMA crossover strategy on AAPL data.
It fetches data for the last 3 years, executes the strategy, and displays the resulting portfolio performance and chart.
This approach is based on a widely used open‑source system, eliminating custom alignment issues.
""")

if st.button("Run Backtrader Simulation"):
    fig = run_backtrader()
    if fig is not None:
        st.pyplot(fig)
