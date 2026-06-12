import pandas as pd
df = pd.read_parquet("data/features/ABB.NS.parquet")
print(df[['Close', 'SMA_200','SMA_150','SMA_50', 'RSI', 'RSI_signal', 'open_close_gap', 'daily_range']].tail(10))
