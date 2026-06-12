import pandas as pd
import numpy as np
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

saved_dir = "saved_models"
features_dir = "data/features"

# Check if model files exist
required_files = ["lgbm.joblib", "xgb.joblib", "cat.joblib", "hgb.joblib", "imputer.joblib", "feature_cols.joblib", "ranked_feature_cols.joblib"]
missing = [f for f in required_files if not os.path.exists(os.path.join(saved_dir, f))]
if missing:
    raise FileNotFoundError(f"Missing model files: {missing}. Please run train_v2.py first.")

# Load models and configurations
print("Loading pre-trained models and configurations...")
trained_lgbm = joblib.load(f"{saved_dir}/lgbm.joblib")
trained_xgb = joblib.load(f"{saved_dir}/xgb.joblib")
trained_cat = joblib.load(f"{saved_dir}/cat.joblib")
trained_hgb = joblib.load(f"{saved_dir}/hgb.joblib")
imputer = joblib.load(f"{saved_dir}/imputer.joblib")
feature_cols = joblib.load(f"{saved_dir}/feature_cols.joblib")
ranked_feature_cols = joblib.load(f"{saved_dir}/ranked_feature_cols.joblib")

def load_parquet(path):
    df = pd.read_parquet(path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def add_features(df):
    df = df.copy()
    df['52w_high'] = df['Close'].rolling(252).max()
    df['52w_low']  = df['Close'].rolling(252).min()

    # All features are ratios/normalized — universal across any stock
    df['dist_SMA50']     = (df['Close'] - df['SMA_50'])  / df['SMA_50']
    df['dist_SMA150']    = (df['Close'] - df['SMA_150']) / df['SMA_150']
    df['dist_SMA200']    = (df['Close'] - df['SMA_200']) / df['SMA_200']
    df['dist_52w_high']  = (df['Close'] - df['52w_high']) / df['52w_high']
    df['dist_52w_low']   = (df['Close'] - df['52w_low'])  / df['52w_low']
    df['SMA150_200_gap'] = (df['SMA_150'] - df['SMA_200']) / df['SMA_200']
    df['SMA50_150_gap']  = (df['SMA_50']  - df['SMA_150']) / df['SMA_150']
    df['SMA200_slope_1m'] = (df['SMA_200'] - df['SMA_200'].shift(22))  / df['SMA_200'].shift(22)
    df['SMA200_slope_3m'] = (df['SMA_200'] - df['SMA_200'].shift(66))  / df['SMA_200'].shift(66)
    df['SMA200_slope_5m'] = (df['SMA_200'] - df['SMA_200'].shift(110)) / df['SMA_200'].shift(110)

    df['RSI_5d_change']   = df['RSI'] - df['RSI'].shift(5)
    df['RSI_10d_change']  = df['RSI'] - df['RSI'].shift(10)
    df['close_5d_return'] = df['Close'].pct_change(5)
    df['close_10d_return']= df['Close'].pct_change(10)
    df['close_20d_return']= df['Close'].pct_change(20)
    df['volume_ratio']    = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()

    return df

# ── Step 1: Load BUY candidates ───────────────────────
buy_df = pd.read_csv("data/qualified_stocks.csv")
files = os.listdir(features_dir)

if len(buy_df) == 0:
    print("\nNo BUY candidates today.")
else:
    print(f"\nProcessing {len(buy_df)} BUY candidates for real-time ranking...")
    
    # Get today's cross-sectional snapshot from ALL stocks
    print("Gathering today's snapshot for relative cross-sectional ranking...")
    today_features = []
    for file in files:
        try:
            df = load_parquet(f"{features_dir}/{file}")
            df = df.sort_index()
            df = add_features(df)
            latest = df.iloc[-1].copy()
            latest['Symbol'] = file.replace(".parquet", "")
            today_features.append(latest[feature_cols + ['Symbol']])
        except:
            pass

    today_df = pd.DataFrame(today_features)

    # Normalize today's features cross-sectionally
    for col in feature_cols:
        today_df[f'{col}_rank'] = today_df[col].rank(pct=True)

    ranked_rows = []
    
    # Run fast model inference
    for _, row in buy_df.iterrows():
        symbol = row['Symbol']
        try:
            stock_row = today_df[today_df['Symbol'] == symbol]
            if stock_row.empty:
                continue

            feat = pd.DataFrame(
                imputer.transform(stock_row[ranked_feature_cols]),
                columns=ranked_feature_cols
            )

            # High-speed prediction using pre-loaded models (sub-millisecond)
            p_lgbm = trained_lgbm.predict_proba(feat)[0][1]
            p_xgb  = trained_xgb.predict_proba(feat)[0][1]
            p_cat  = trained_cat.predict(feat)[0]
            p_hgb  = trained_hgb.predict(feat)[0]
            final  = (p_lgbm + p_xgb + p_cat + p_hgb) / 4

            ranked_rows.append({
                'Symbol'      : symbol,
                'Close'       : row['Close'],
                'RSI'         : row['RSI'],
                'LGBM_Score'  : round(p_lgbm, 4),
                'XGB_Score'   : round(p_xgb,  4),
                'CAT_Score'   : round(p_cat,  4),
                'HGB_Score'   : round(p_hgb,  4),
                'Final_Score' : round(final,  4),
                'ACTION'      : 'BUY'
            })

        except Exception as e:
            print(f"  Ranking failed for {symbol}: {e}")

    ranked_df = pd.DataFrame(ranked_rows)
    if not ranked_df.empty:
        ranked_df = ranked_df.sort_values('Final_Score', ascending=False).reset_index(drop=True)
        ranked_df['Rank'] = ranked_df.index + 1

        print(f"\n{'='*55}")
        print(f"  TOP BUY CANDIDATES TODAY (Universal Ensemble - Offline Trained)")
        print(f"{'='*55}")
        print(ranked_df[['Rank', 'Symbol', 'Close', 'RSI', 'Final_Score']].to_string(index=False))

        ranked_df.to_csv("data/ranked_buy_stocks.csv", index=False)
        print("\nSaved to data/ranked_buy_stocks.csv")
    else:
        print("\nNo stocks ranked successfully.")
