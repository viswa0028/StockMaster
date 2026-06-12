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
import joblib

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

print("Loading historical parquet files...")
for i, file in enumerate(files):
    try:
        df = load_parquet(f"{features_dir}/{file}")
        df = df.sort_index()
        df = add_features(df)
        df['Symbol'] = file.replace(".parquet", "")
        all_data.append(df)
        if i % 50 == 0:
            print(f"  Loaded {i}/{len(files)}...")
    except Exception as e:
        print(f"  Failed loading: {file} — {e}")

master_df = pd.concat(all_data)
master_df = master_df.sort_index()

# ── Step 2: Cross-sectional rank target ──────────────
print("\nBuilding cross-sectional rank labels...")
master_df['target'] = master_df.groupby(level=0)['future_return'].transform(
    lambda x: (x.rank(pct=True) >= 0.80).astype(int)
)

# ── Step 3: Cross-sectional normalization ─────────────
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
        lambda x: x.rank(pct=True)
    )

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
print("\nEvaluating model via Walk-Forward Validation...")
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

    # Use probability scores for ranking across ALL models
    p_lgbm = lgbm.predict_proba(X_test)[:, 1]
    p_xgb  = xgb_m.predict_proba(X_test)[:, 1]
    p_cat  = cat.predict_proba(X_test)[:, 1]
    p_hgb  = hgb.predict_proba(X_test)[:, 1]
    
    p_ens  = (p_lgbm + p_xgb + p_cat + p_hgb) / 4

    from sklearn.metrics import roc_auc_score
    score = roc_auc_score(y_test, p_ens)
    fold_scores['lgbm'].append(roc_auc_score(y_test, p_lgbm))
    fold_scores['xgb'].append(roc_auc_score(y_test, p_xgb))
    fold_scores['cat'].append(roc_auc_score(y_test, p_cat))
    fold_scores['hgb'].append(roc_auc_score(y_test, p_hgb))
    fold_scores['ensemble'].append(score)

    print(f"Fold {fold+1} — Ensemble AUC: {score:.4f}")

    if score > best_score:
        best_score   = score
        trained_lgbm = lgbm
        trained_xgb  = xgb_m
        trained_cat  = cat
        trained_hgb  = hgb

print(f"\n── Average AUC Scores ──────────────────────────")
for name, s in fold_scores.items():
    print(f"{name.upper():10} : {np.mean(s):.4f}")
print(f"Best Ensemble AUC   : {best_score:.4f}")

# ── Step 6: Save Models and Imputer ───────────────────
print("\nSaving best models and pipeline parameters to disk...")
saved_dir = "saved_models"
os.makedirs(saved_dir, exist_ok=True)

joblib.dump(trained_lgbm, f"{saved_dir}/lgbm.joblib")
joblib.dump(trained_xgb, f"{saved_dir}/xgb.joblib")
joblib.dump(trained_cat, f"{saved_dir}/cat.joblib")
joblib.dump(trained_hgb, f"{saved_dir}/hgb.joblib")
joblib.dump(imputer, f"{saved_dir}/imputer.joblib")
joblib.dump(feature_cols, f"{saved_dir}/feature_cols.joblib")
joblib.dump(ranked_feature_cols, f"{saved_dir}/ranked_feature_cols.joblib")

print("Done! All models successfully saved.")
