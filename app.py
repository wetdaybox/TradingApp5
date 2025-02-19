import streamlit as st
import pandas as pd
import numpy as np
import json
import asyncio
from datetime import datetime
from binance import AsyncClient, BinanceSocketManager

st.title("💰 Free Crypto Trading Signals")

async def main():
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    ts = bm.trade_socket('BTCUSDT')
    
    async with ts as tscm:
        while True:
            res = await tscm.recv()
            st.write(f"""
            **BTC/USDT Live Data**
            - Price: ${float(res['p']):.2f}
            - Quantity: {float(res['q']):.4f}
            - Time: {datetime.fromtimestamp(res['T']/1000)}
            """)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
