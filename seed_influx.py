import pandas as pd
import os
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

# ── Configuration ──────────────────────────────────────
TOKEN = "my-super-secret-auth-token"
ORG = "stockmaster"
BUCKET = "market_data"
URL = "http://localhost:8086"

features_dir = "data/features"
files = [f for f in os.listdir(features_dir) if f.endswith('.parquet')]

# ── Connect to InfluxDB ────────────────────────────────
client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

print(f"Connected to InfluxDB. Found {len(files)} ticker files to seed.")

def seed_ticker(file_name):
    ticker = file_name.replace(".parquet", "")
    file_path = os.path.join(features_dir, file_name)
    
    df = pd.read_parquet(file_path)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"Skipping {ticker}: Index is not DatetimeIndex.")
        return
        
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[[c for c in cols_to_keep if c in df.columns]]
    
    try:
        # THE MAGIC FIX: We set the measurement name exactly to the ticker (e.g., "AAPL")
        write_api.write(
            bucket=BUCKET, 
            record=df, 
            data_frame_measurement_name=ticker 
        )
        print(f"Successfully seeded {ticker} ({len(df)} records).")
    except Exception as e:
        print(f"Failed to write {ticker}: {e}")

# ── Execute Seeding ────────────────────────────────────
for file in files:
    seed_ticker(file)

client.close()
print("Database seeding complete!")