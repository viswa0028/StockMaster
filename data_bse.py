import yfinance as yf
import pandas as pd
import os, time

# Step 1: Get BSE 500 list
url = "https://archives.nseindia.com/content/indices/ind_bse500list.csv"
bse500 = pd.read_csv(url)
symbols = [s + ".BO" for s in bse500['Symbol'].tolist()]

os.makedirs("data/bse_raw", exist_ok=True)

# Step 2: Download in batches of 50
batch_size = 50
for i in range(0, len(symbols), batch_size):
    batch = symbols[i:i+batch_size]
    print(f"Downloading batch {i//batch_size + 1}...")
    
    for symbol in batch:
        try:
            df = yf.download(symbol, start="2015-01-01",
                             end="2024-01-01", progress=False)
            if len(df) > 100:
                df.to_parquet(f"data/bse_raw/{symbol}.parquet")
        except Exception as e:
            print(f"  Failed: {symbol}")
    
    time.sleep(2)

print("Done!")