import yfinance as yf
import pandas as pd
import os, time
from datetime import date

# Get Nifty 500 list
url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
nifty500 = pd.read_csv(url)
symbols = [s + ".NS" for s in nifty500['Symbol'].tolist()]

os.makedirs("data/raw", exist_ok=True)

batch_size = 50
for i in range(0, len(symbols), batch_size):
    batch = symbols[i:i+batch_size]
    print(f"Downloading batch {i//batch_size + 1}...")

    for symbol in batch:
        try:
            df = yf.download(
                symbol,
                start="2015-01-01",
                end=date.today().strftime("%Y-%m-%d"),  # ← always today
                progress=False
            )
            if len(df) > 100:
                df.to_parquet(f"data/raw/{symbol}.parquet")
        except Exception as e:
            print(f"  Failed: {symbol}")

    time.sleep(2)

print("Done!")