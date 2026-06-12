import pandas as pd
import os

input_dir = "data/raw"
output_dir = "data/features"
os.makedirs(output_dir, exist_ok=True)

files = os.listdir(input_dir)
print(f"Total stocks: {len(files)}")

for i, file in enumerate(files):
    try:
        df = pd.read_parquet(f"{input_dir}/{file}")
        df.dropna(inplace=True)

        # ── Moving Averages ──────────────────────────────
        df['SMA_10']  = df['Close'].rolling(window=10).mean()
        df['SMA_20']  = df['Close'].rolling(window=20).mean()
        df['SMA_50']  = df['Close'].rolling(window=50).mean()
        df['SMA_150'] = df['Close'].rolling(window=150).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # ── Price Features (Open, High, Low matter) ──────
        df['daily_range']    = df['High'] - df['Low']          # volatility of the day
        df['open_close_gap'] = df['Close'] - df['Open']        # bullish if positive
        df['high_close_gap'] = df['High'] - df['Close']        # selling pressure near high
        df['low_close_gap']  = df['Close'] - df['Low']         # buying support near low
        df['gap_up_down']    = df['Open'] - df['Close'].shift(1)  # gap from previous close

        # ── RSI (14-day) ─────────────────────────────────
        delta = df['Close'].diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs        = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # ── RSI Signal (client rule) ──────────────────────
        # RSI > 55 → Buy signal (1), RSI < 45 → Sell signal (-1), else Hold (0)
        df['RSI_signal'] = 0
        df.loc[df['RSI'] > 55, 'RSI_signal'] =  1
        df.loc[df['RSI'] < 45, 'RSI_signal'] = -1

        # ── Drop rows where indicators aren't ready ───────
        df.dropna(inplace=True)

        df.to_parquet(f"{output_dir}/{file}")

        if i % 50 == 0:
            print(f"Processed {i}/{len(files)}...")

    except Exception as e:
        print(f"Failed: {file} — {e}")

print("Completed")