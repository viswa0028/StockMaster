import pandas as pd
import os

features_dir = "data/features"
files = os.listdir(features_dir)

results = []

def load_parquet(path):
    df = pd.read_parquet(path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

for file in files:
    try:
        df = load_parquet(f"{features_dir}/{file}")
        df = df.sort_index()

        latest = df.iloc[-1]
        symbol = file.replace(".parquet", "")

        last_252 = df['Close'].tail(252)
        week52_high = last_252.max()
        week52_low  = last_252.min()

        c1    = latest['Close'] > latest['SMA_150'] and latest['Close'] > latest['SMA_200']
        c2    = latest['SMA_150'] > latest['SMA_200']
        c3    = latest['SMA_200'] > df['SMA_200'].iloc[-22] if len(df) >= 22 else False
        c4    = latest['SMA_50'] > latest['SMA_150'] and latest['SMA_50'] > latest['SMA_200']
        c5    = latest['Close'] >= week52_low * 1.30
        c6    = latest['Close'] <= week52_high * 1.25

        # ── RSI Signals ──────────────────────────────────
        rsi        = latest['RSI']
        rsi_buy    = rsi > 55   # buy condition
        rsi_sell   = rsi < 45   # sell condition

        all_conditions = c1 and c2 and c3 and c4 and c5 and c6 and rsi_buy

        # ── Action label ─────────────────────────────────
        if all_conditions:
            action = "BUY"
        elif rsi_sell:
            action = "SELL"
        else:
            action = "HOLD"

        results.append({
            'Symbol'        : symbol,
            'Close'         : round(latest['Close'], 2),
            'SMA_50'        : round(latest['SMA_50'], 2),
            'SMA_150'       : round(latest['SMA_150'], 2),
            'SMA_200'       : round(latest['SMA_200'], 2),
            'RSI'           : round(rsi, 2),
            '52w_High'      : round(week52_high, 2),
            '52w_Low'       : round(week52_low, 2),
            'C1_Price>MA'   : c1,
            'C2_150>200'    : c2,
            'C3_200Trend'   : c3,
            'C4_50>150&200' : c4,
            'C5_30%>52wL'   : c5,
            'C6_25%<52wH'   : c6,
            'RSI_BUY'       : rsi_buy,
            'RSI_SELL'      : rsi_sell,
            'ACTION'        : action,
            'PASSED'        : all_conditions
        })

    except Exception as e:
        print(f"Failed: {file} — {e}")

results_df = pd.DataFrame(results)
buy_df  = results_df[results_df['ACTION'] == 'BUY'].reset_index(drop=True)
sell_df = results_df[results_df['ACTION'] == 'SELL'].reset_index(drop=True)

print(f"\nTotal screened : {len(results_df)}")
print(f"BUY signals    : {len(buy_df)}")
print(f"SELL signals   : {len(sell_df)}")

results_df.to_csv("data/screener_results.csv", index=False)
buy_df.to_csv("data/qualified_stocks.csv", index=False)
sell_df.to_csv("data/sell_signals.csv", index=False)
print("\nSaved screener_results, qualified_stocks, sell_signals")