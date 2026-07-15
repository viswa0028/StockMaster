import pandas as pd
import numpy as np
import os
import joblib
import warnings
from datetime import date as _date
from state_manager_redis import MarketStateManager
from daily_sync import run_daily_sync

warnings.filterwarnings('ignore')

saved_dir = "saved_models"
features_dir = "data/features"

# Check if model files exist
required_files = ["lgbm.joblib", "xgb.joblib", "cat.joblib", "hgb.joblib", "imputer.joblib", "feature_cols.joblib", "ranked_feature_cols.joblib"]
missing = [f for f in required_files if not os.path.exists(os.path.join(saved_dir, f))]
if missing:
    raise FileNotFoundError(f"Missing model files: {missing}. Please run train_v2.py first.")

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

def sync_data_if_needed(target_date_str, _files=None):
    """
    Delegate to the canonical daily_sync module.
    Downloads missing daily OHLCV data from yfinance, updates parquet files,
    writes to InfluxDB, and refreshes Redis macro context.
    """
    target = _date.fromisoformat(target_date_str)
    run_daily_sync(target_date=target)


def run_historical_playback_for_date(target_date_str="2019-11-11"):
    print(f"\n🚀 Running Historical Playback for Date: {target_date_str}")
    target_date = pd.to_datetime(target_date_str)
    
    files = [f for f in os.listdir(features_dir) if f.endswith('.parquet')]
    print(f"Scanning {len(files)} ticker files...")
    
    sync_data_if_needed(target_date_str, files)
    
    snapshot_rows = []
    minervini_passed = []
    
    for file in files:
        symbol = file.replace(".parquet", "")
        try:
            df = load_parquet(os.path.join(features_dir, file))
            df = df.sort_index()
            df = add_features(df)
            
            # Find the specific row for target_date
            if target_date in df.index:
                actual_date = target_date
            else:
                # Look back up to 14 days to find the most recent available data
                past_dates = df.index[df.index <= target_date]
                if len(past_dates) > 0 and (target_date - past_dates[-1]).days <= 14:
                    actual_date = past_dates[-1]
                else:
                    continue
                
            idx = df.index.get_loc(actual_date)
            latest = df.iloc[idx]
            
            # Check Minervini trend conditions
            c1 = latest['Close'] > latest['SMA_150'] and latest['Close'] > latest['SMA_200']
            c2 = latest['SMA_150'] > latest['SMA_200']
            c3 = latest['SMA_200'] > df['SMA_200'].iloc[idx-22] if idx >= 22 else False
            c4 = latest['SMA_50'] > latest['SMA_150'] and latest['SMA_50'] > latest['SMA_200']
            
            # 52w High / Low calculation as of this date
            last_252 = df['Close'].iloc[max(0, idx-251):idx+1]
            week52_high = last_252.max()
            week52_low  = last_252.min()
            
            c5 = latest['Close'] >= week52_low * 1.30
            c6 = latest['Close'] <= week52_high * 1.25

            ma_values = [latest["SMA_20"], latest["SMA_50"], latest["SMA_150"], latest["SMA_200"]]
            ma_spread_pct = (max(ma_values) - min(ma_values)) / latest["Close"] * 100
            c7 = ma_spread_pct <= 10.0
            rsi_buy = latest["RSI"] > 55

            passed_macro = c1 and c2 and c3 and c4 and c5 and c6 #and rsi_buy #and c7
            
            features_dict = latest[feature_cols].to_dict()
            features_dict['Symbol'] = symbol
            features_dict['Close'] = latest['Close']
            features_dict['RSI'] = latest['RSI']
            features_dict['Passed_Macro'] = passed_macro
            
            snapshot_rows.append(features_dict)
            
        except Exception as e:
            pass
            
    if not snapshot_rows:
        print(f"No stock data found for the date {target_date_str}. Make sure it is a trading day.")
        return
        
    snapshot_df = pd.DataFrame(snapshot_rows)
    
    # Calculate cross-sectional ranks for this specific historical day
    for col in feature_cols:
        snapshot_df[f'{col}_rank'] = snapshot_df[col].rank(pct=True)
        
    # Filter only those that passed Minervini macro conditions
    passed_df = snapshot_df[snapshot_df['Passed_Macro'] == True].reset_index(drop=True)
    print(f"Total stocks scanned: {len(snapshot_df)} | Passed Minervini Trend Rules: {len(passed_df)}")
    
    if len(passed_df) == 0:
        print("No stocks passed the macro filters on this day.")
        return
        
    # Evaluate ML models
    ranked_rows = []
    for _, row in passed_df.iterrows():
        symbol = row['Symbol']
        try:
            feat = pd.DataFrame([row[ranked_feature_cols]], columns=ranked_feature_cols)
            feat_imputed = pd.DataFrame(imputer.transform(feat), columns=ranked_feature_cols)
            
            p_lgbm = trained_lgbm.predict_proba(feat_imputed)[0][1]
            p_xgb  = trained_xgb.predict_proba(feat_imputed)[0][1]
            p_cat  = trained_cat.predict_proba(feat_imputed)[0][1]
            p_hgb  = trained_hgb.predict_proba(feat_imputed)[0][1]
            final  = (p_lgbm + p_xgb + p_cat + p_hgb) / 4
            
            ranked_rows.append({
                'Symbol': symbol,
                'Close': round(row['Close'], 2),
                'RSI': round(row['RSI'], 2),
                'LGBM_Score': round(p_lgbm, 4),
                'XGB_Score': round(p_xgb, 4),
                'CAT_Score': round(p_cat, 4),
                'HGB_Score': round(p_hgb, 4),
                'Final_Score': round(final, 4)
            })
        except Exception as e:
            print(f"Prediction failed for {symbol}: {e}")
            
    ranked_df = pd.DataFrame(ranked_rows)
    if not ranked_df.empty:
        ranked_df = ranked_df.sort_values('Final_Score', ascending=False).reset_index(drop=True)
        ranked_df['Rank'] = ranked_df.index + 1
        
        print(f"\n=======================================================")
        print(f"  TOP BUY CANDIDATES ON {target_date_str} (Ensemble Ranked)")
        print(f"=======================================================")
        print(ranked_df[['Rank', 'Symbol', 'Close', 'RSI', 'Final_Score']].head(50).to_string(index=False))
        
        # Save output
        out_file = f"data/ranked_buy_{target_date_str}.csv"
        ranked_df.to_csv(out_file, index=False)
        print(f"\nResults saved to {out_file}")
    else:
        print("No stocks ranked.")

if __name__ == "__main__":
    # Play back a historical day
    run_historical_playback_for_date("2026-06-17")