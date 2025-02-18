import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

ticker = "AAPL"
years = 5
end = datetime.today()
start = end - timedelta(days=years * 365)

# Download data
df = yf.download(ticker, start=start, end=end, progress=False)
if df.empty:
    print("No data found!")
else:
    df = df[['Close', 'Volume', 'High', 'Low']].rename(columns={'Close': 'Price'})
    full_index = pd.date_range(start=start, end=end, freq='B')
    df = df.reindex(full_index).ffill().dropna()
    print("Stage 1: Data shape:", df.shape)
    print("Stage 1: Data head:")
    print(df.head())
