# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import xgboost as xgb
# from catboost import CatBoostRegressor
# from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.impute import SimpleImputer
# import os
# import warnings
# warnings.filterwarnings('ignore')

# features_dir = "data/features"
# files = os.listdir(features_dir)

# def load_parquet(path):
#     df = pd.read_parquet(path)
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = df.columns.get_level_values(0)
#     df = df.loc[:, ~df.columns.duplicated()]
#     return df

# def add_features(df):
#     df = df.copy()
#     df['52w_high'] = df['Close'].rolling(252).max()
#     df['52w_low']  = df['Close'].rolling(252).min()

#     df['dist_SMA50']     = (df['Close'] - df['SMA_50'])  / df['SMA_50']
#     df['dist_SMA150']    = (df['Close'] - df['SMA_150']) / df['SMA_150']
#     df['dist_SMA200']    = (df['Close'] - df['SMA_200']) / df['SMA_200']
#     df['dist_52w_high']  = (df['Close'] - df['52w_high']) / df['52w_high']
#     df['dist_52w_low']   = (df['Close'] - df['52w_low'])  / df['52w_low']
#     df['SMA150_200_gap'] = (df['SMA_150'] - df['SMA_200']) / df['SMA_200']
#     df['SMA50_150_gap']  = (df['SMA_50']  - df['SMA_150']) / df['SMA_150']
#     df['SMA200_slope']   = (df['SMA_200'] - df['SMA_200'].shift(22)) / df['SMA_200'].shift(22)

#     # ── Momentum features (how indicators changed over time) ──
#     df['RSI_5d_change']      = df['RSI'] - df['RSI'].shift(5)
#     df['RSI_10d_change']     = df['RSI'] - df['RSI'].shift(10)
#     df['close_5d_return']    = df['Close'].pct_change(5)
#     df['close_10d_return']   = df['Close'].pct_change(10)
#     df['close_20d_return']   = df['Close'].pct_change(20)
#     df['volume_5d_avg']      = df['Volume'].rolling(5).mean()
#     df['volume_20d_avg']     = df['Volume'].rolling(20).mean()
#     df['volume_ratio']       = df['volume_5d_avg'] / df['volume_20d_avg']  # volume surge
#     df['dist_SMA50_5d_ago']  = df['dist_SMA50'].shift(5)
#     df['SMA200_slope_1m']    = (df['SMA_200'] - df['SMA_200'].shift(22)) / df['SMA_200'].shift(22)
#     df['SMA200_slope_3m']    = (df['SMA_200'] - df['SMA_200'].shift(66)) / df['SMA_200'].shift(66)
#     df['SMA200_slope_5m']    = (df['SMA_200'] - df['SMA_200'].shift(110)) / df['SMA_200'].shift(110)

#     # ── Target: forward 10-day return ─────────────────────
#     df['future_return'] = df['Close'].shift(-10) / df['Close'] - 1

#     return df

# # ── Step 1: Build full dataset with all history ───────
# all_data = []

# for i, file in enumerate(files):
#     try:
#         df = load_parquet(f"{features_dir}/{file}")
#         df = df.sort_index()
#         df = add_features(df)
#         df['Symbol'] = file.replace(".parquet", "")
#         all_data.append(df)
#         if i % 50 == 0:
#             print(f"Loaded {i}/{len(files)}...")
#     except Exception as e:
#         print(f"Failed: {file} — {e}")

# master_df = pd.concat(all_data)
# master_df.dropna(inplace=True)
# master_df = master_df.sort_index()  # sort by date across all stocks

# print(f"\nTotal training samples : {len(master_df)}")
# print(f"Date range             : {master_df.index.min()} → {master_df.index.max()}")

# # ── Step 2: Features ──────────────────────────────────
# feature_cols = [
#     'RSI', 'SMA_10', 'SMA_20',
#     'dist_SMA50', 'dist_SMA150', 'dist_SMA200',
#     'dist_52w_high', 'dist_52w_low',
#     'SMA150_200_gap', 'SMA50_150_gap',
#     'SMA200_slope',
#     'daily_range', 'open_close_gap',
#     'high_close_gap', 'low_close_gap', 'gap_up_down',
#     'RSI_5d_change', 'RSI_10d_change',
#     'close_5d_return', 'close_10d_return', 'close_20d_return',
#     'volume_ratio',
#     'dist_SMA50_5d_ago',
#     'SMA200_slope_1m', 'SMA200_slope_3m', 'SMA200_slope_5m'
# ]

# feature_cols = [c for c in feature_cols if c in master_df.columns]

# X = master_df[feature_cols]
# y = master_df['future_return']

# print(f"Features used          : {len(feature_cols)}")

# # ── Step 3: Impute any remaining NaNs ────────────────
# imputer = SimpleImputer(strategy='median')
# X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

# # ── Step 4: Walk-Forward CV ───────────────────────────
# tscv = TimeSeriesSplit(n_splits=5)
# fold_scores = {'lgbm': [], 'xgb': [], 'cat': [], 'hgb': [], 'ensemble': []}

# trained_lgbm = trained_xgb = trained_cat = trained_hgb = None
# best_score = -999

# for fold, (train_idx, test_idx) in enumerate(tscv.split(X_imputed)):
#     X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#     lgbm = lgb.LGBMRegressor(
#         n_estimators=1000, learning_rate=0.03,
#         max_depth=6, num_leaves=31,
#         subsample=0.8, colsample_bytree=0.8,
#         random_state=42, verbose=-1
#     )

#     xgb_m = xgb.XGBRegressor(
#         n_estimators=1000, learning_rate=0.03,
#         max_depth=6, subsample=0.8,
#         colsample_bytree=0.8, random_state=42, verbosity=0
#     )

#     cat = CatBoostRegressor(
#         iterations=1000, learning_rate=0.03,
#         depth=6, random_seed=42, verbose=0
#     )

#     hgb = HistGradientBoostingRegressor(
#         max_iter=1000, learning_rate=0.03,
#         max_depth=6, random_state=42
#     )

#     lgbm.fit(X_train, y_train)
#     xgb_m.fit(X_train, y_train)
#     cat.fit(X_train, y_train)
#     hgb.fit(X_train, y_train)

#     p_lgbm = lgbm.predict(X_test)
#     p_xgb  = xgb_m.predict(X_test)
#     p_cat  = cat.predict(X_test)
#     p_hgb  = hgb.predict(X_test)
#     p_ens  = (p_lgbm + p_xgb + p_cat + p_hgb) / 4

#     fold_scores['lgbm'].append(np.corrcoef(p_lgbm, y_test)[0,1])
#     fold_scores['xgb'].append(np.corrcoef(p_xgb,  y_test)[0,1])
#     fold_scores['cat'].append(np.corrcoef(p_cat,  y_test)[0,1])
#     fold_scores['hgb'].append(np.corrcoef(p_hgb,  y_test)[0,1])
#     fold_scores['ensemble'].append(np.corrcoef(p_ens, y_test)[0,1])

#     ens_score = fold_scores['ensemble'][-1]
#     print(f"Fold {fold+1} — LGBM: {fold_scores['lgbm'][-1]:.4f} | "
#           f"XGB: {fold_scores['xgb'][-1]:.4f} | "
#           f"CAT: {fold_scores['cat'][-1]:.4f} | "
#           f"HGB: {fold_scores['hgb'][-1]:.4f} | "
#           f"Ensemble: {ens_score:.4f}")

#     if ens_score > best_score:
#         best_score   = ens_score
#         trained_lgbm = lgbm
#         trained_xgb  = xgb_m
#         trained_cat  = cat
#         trained_hgb  = hgb

# print(f"\n── Average Scores ──────────────────────────────")
# for name, s in fold_scores.items():
#     print(f"{name.upper():10} : {np.mean(s):.4f}")
# print(f"Best Ensemble Score : {best_score:.4f}")

# # ── Step 5: Rank today's BUY candidates ──────────────
# buy_df = pd.read_csv("data/qualified_stocks.csv")

# if len(buy_df) == 0:
#     print("\nNo BUY candidates today.")
# else:
#     ranked_rows = []

#     for _, row in buy_df.iterrows():
#         symbol = row['Symbol']
#         try:
#             df = load_parquet(f"{features_dir}/{symbol}.parquet")
#             df = df.sort_index()
#             df = add_features(df)

#             latest = df.iloc[-1]
#             feat   = pd.DataFrame([latest[feature_cols]])
#             feat   = pd.DataFrame(imputer.transform(feat), columns=feature_cols)

#             s_lgbm = trained_lgbm.predict(feat)[0]
#             s_xgb  = trained_xgb.predict(feat)[0]
#             s_cat  = trained_cat.predict(feat)[0]
#             s_hgb  = trained_hgb.predict(feat)[0]
#             final  = (s_lgbm + s_xgb + s_cat + s_hgb) / 4

#             ranked_rows.append({
#                 'Symbol'      : symbol,
#                 'Close'       : row['Close'],
#                 'RSI'         : row['RSI'],
#                 'LGBM_Score'  : round(s_lgbm, 4),
#                 'XGB_Score'   : round(s_xgb,  4),
#                 'CAT_Score'   : round(s_cat,  4),
#                 'HGB_Score'   : round(s_hgb,  4),
#                 'Final_Score' : round(final,  4),
#                 'ACTION'      : 'BUY'
#             })

#         except Exception as e:
#             print(f"Ranking failed: {symbol} — {e}")

#     ranked_df = pd.DataFrame(ranked_rows)
#     ranked_df = ranked_df.sort_values('Final_Score', ascending=False).reset_index(drop=True)
#     ranked_df['Rank'] = ranked_df.index + 1

#     print(f"\n{'='*55}")
#     print(f"  TOP BUY CANDIDATES TODAY (Ensemble Ranked)")
#     print(f"{'='*55}")
#     print(ranked_df[['Rank', 'Symbol', 'Close', 'RSI', 'Final_Score']].to_string(index=False))

#     ranked_df.to_csv("data/ranked_buy_stocks.csv", index=False)
#     print("\nSaved to data/ranked_buy_stocks.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings('ignore')

features_dir = "data/features"
files = os.listdir(features_dir)

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

    # Forward return for target
    df['future_return'] = df['Close'].shift(-10) / df['Close'] - 1

    return df

# ── Step 1: Load all stocks ───────────────────────────
all_data = []

for i, file in enumerate(files):
    try:
        df = load_parquet(f"{features_dir}/{file}")
        df = df.sort_index()
        df = add_features(df)
        df['Symbol'] = file.replace(".parquet", "")
        all_data.append(df)
        if i % 50 == 0:
            print(f"Loaded {i}/{len(files)}...")
    except Exception as e:
        print(f"Failed loading: {file} — {e}")

master_df = pd.concat(all_data)
master_df = master_df.sort_index()

# ── Step 2: Cross-sectional rank target ──────────────
# For each day, rank all stocks by future_return
# Top 20% → label 1 (strong buy), rest → label 0
# This makes the model learn RELATIVE strength, not absolute returns
print("\nBuilding cross-sectional rank labels...")
master_df['target'] = master_df.groupby(level=0)['future_return'].transform(
    lambda x: (x.rank(pct=True) >= 0.80).astype(int)
)

# ── Step 3: Cross-sectional normalization ─────────────
# Normalize each feature per day across all stocks
# So RSI=60 is judged relative to all other stocks that day
feature_cols = [
    'RSI', 'dist_SMA50', 'dist_SMA150', 'dist_SMA200',
    'dist_52w_high', 'dist_52w_low',
    'SMA150_200_gap', 'SMA50_150_gap',
    'SMA200_slope_1m', 'SMA200_slope_3m', 'SMA200_slope_5m',
    'RSI_5d_change', 'RSI_10d_change',
    'close_5d_return', 'close_10d_return', 'close_20d_return',
    'volume_ratio',
    'daily_range', 'open_close_gap', 'high_close_gap',
    'low_close_gap', 'gap_up_down'
]

feature_cols = [c for c in feature_cols if c in master_df.columns]

print("Normalizing features cross-sectionally per day...")
for col in feature_cols:
    master_df[f'{col}_rank'] = master_df.groupby(level=0)[col].transform(
        lambda x: x.rank(pct=True)  # converts to 0-1 percentile rank per day
    )

# Use ranked features — these are truly universal
ranked_feature_cols = [f'{c}_rank' for c in feature_cols]

master_df.dropna(subset=ranked_feature_cols + ['target'], inplace=True)

X = master_df[ranked_feature_cols]
y = master_df['target']

print(f"\nTotal training samples : {len(master_df)}")
print(f"Positive labels (top20%): {y.sum()} ({y.mean()*100:.1f}%)")
print(f"Features used          : {len(ranked_feature_cols)}")

# ── Step 4: Imputer ───────────────────────────────────
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=ranked_feature_cols,
    index=X.index
)

# ── Step 5: Walk-Forward CV ───────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
fold_scores = {'lgbm': [], 'xgb': [], 'cat': [], 'hgb': [], 'ensemble': []}
trained_lgbm = trained_xgb = trained_cat = trained_hgb = None
best_score = -999

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_imputed)):
    X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    lgbm = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03,
        max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
        class_weight='balanced'
    )
    xgb_m = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.03,
        max_depth=6, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        verbosity=0, scale_pos_weight=4
    )
    cat = CatBoostClassifier(
        iterations=1000, learning_rate=0.03,
        depth=6, random_seed=42, verbose=0
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=1000, learning_rate=0.03,
        max_depth=6, random_state=42
    )
    lgbm.fit(X_train, y_train)
    xgb_m.fit(X_train, y_train)
    cat.fit(X_train, y_train)
    hgb.fit(X_train, y_train)

    # Use probability scores for ranking
    p_lgbm = lgbm.predict_proba(X_test)[:, 1]
    p_xgb  = xgb_m.predict_proba(X_test)[:, 1]
    p_cat  = cat.predict(X_test)
    p_hgb  = hgb.predict(X_test)
    p_ens  = (p_lgbm + p_xgb + p_cat + p_hgb) / 4

    # Measure how well ensemble identifies top 20%
    from sklearn.metrics import roc_auc_score
    score = roc_auc_score(y_test, p_ens)
    fold_scores['lgbm'].append(roc_auc_score(y_test, p_lgbm))
    fold_scores['xgb'].append(roc_auc_score(y_test, p_xgb))
    fold_scores['cat'].append(roc_auc_score(y_test, p_cat))
    fold_scores['hgb'].append(roc_auc_score(y_test, p_hgb))
    fold_scores['ensemble'].append(score)

    print(f"Fold {fold+1} — LGBM: {fold_scores['lgbm'][-1]:.4f} | "
          f"XGB: {fold_scores['xgb'][-1]:.4f} | "
          f"CAT: {fold_scores['cat'][-1]:.4f} | "
          f"HGB: {fold_scores['hgb'][-1]:.4f} | "
          f"Ensemble: {score:.4f}")

    if score > best_score:
        best_score   = score
        trained_lgbm = lgbm
        trained_xgb  = xgb_m
        trained_cat  = cat
        trained_hgb  = hgb

print(f"\n── Average AUC Scores ──────────────────────────")
for name, s in fold_scores.items():
    print(f"{name.upper():10} : {np.mean(s):.4f}")
print(f"\nNote: AUC > 0.55 is good, > 0.60 is strong for stocks")
print(f"Best Ensemble AUC : {best_score:.4f}")

# ── Step 6: Rank today's BUY candidates ──────────────
buy_df = pd.read_csv("data/qualified_stocks.csv")

if len(buy_df) == 0:
    print("\nNo BUY candidates today.")
else:
    ranked_rows = []

    # Get today's cross-sectional ranks from ALL stocks
    # This is needed to normalize a new stock the same way
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

    # Cross-sectional rank today's features across all available stocks
    for col in feature_cols:
        today_df[f'{col}_rank'] = today_df[col].rank(pct=True)

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
            print(f"Ranking failed: {symbol} — {e}")

    ranked_df = pd.DataFrame(ranked_rows)
    ranked_df = ranked_df.sort_values('Final_Score', ascending=False).reset_index(drop=True)
    ranked_df['Rank'] = ranked_df.index + 1

    print(f"\n{'='*55}")
    print(f"  TOP BUY CANDIDATES TODAY (Universal Ensemble)")
    print(f"{'='*55}")
    print(ranked_df[['Rank', 'Symbol', 'Close', 'RSI', 'Final_Score']].to_string(index=False))

    ranked_df.to_csv("data/ranked_buy_stocks.csv", index=False)
    print("\nSaved to data/ranked_buy_stocks.csv")