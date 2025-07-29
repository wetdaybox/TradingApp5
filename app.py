import requests

def fetch_ticker(symbol):
    """Fetch ticker data for a given symbol from Crypto.com API."""
    url = f"https://api.crypto.com/v1/ticker?symbol={symbol}"
    response = requests.get(url, timeout=10)
    data = response.json()
    if data.get('code') != '0':
        raise RuntimeError(f"API error for {symbol}: {data.get('msg')}")
    return data['data']

def main():
    try:
        # Fetch BTC/USDT ticker to get price and 24h change
        btc_ticker = fetch_ticker("btcusdt")
        btc_price = float(btc_ticker["last"])
        # 'rose' is 24h change as a decimal (e.g. 0.01 = +1%), convert to percentage
        btc_change_pct = float(btc_ticker["rose"]) * 100

        print(f"BTC/USDT 24h change: {btc_change_pct:.2f}%")
        if btc_change_pct >= 0.82:
            print("TRIGGER: BTC rose ≥0.82% in 24h.")
            # Determine range width based on rise magnitude
            if btc_change_pct <= 4.19:
                range_pct = 7.22
            else:
                range_pct = 13.9
            # Fetch current XRP/BTC price
            xrp_ticker = fetch_ticker("xrpbtc")
            xrp_btc_price = float(xrp_ticker["last"])
            top_price = xrp_btc_price
            bottom_price = top_price * (1 - range_pct/100)
            print(f"Recommended grid range: TOP = {top_price:.8f} BTC, BOTTOM = {bottom_price:.8f} BTC ({range_pct}% below).")
            
            # Ask user for investment and grid count
            try:
                investment = float(input("Enter total investment amount (in XRP): "))
            except:
                investment = None
            try:
                num_grids = int(input("Enter number of grid levels: "))
            except:
                num_grids = None
            
            if num_grids and num_grids > 0:
                grid_size = (top_price - bottom_price) / num_grids
                print(f"Grid size for {num_grids} grids: {(grid_size):.8f} BTC per grid level.")
        else:
            print("No adjustment needed (BTC rise < 0.82%).")
    except Exception as e:
        print("Error fetching price data:", e)

if __name__ == "__main__":
    main()
