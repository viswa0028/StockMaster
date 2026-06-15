"""
live_trading_api.py
-------------------
FastAPI application for StockMaster real-time trading signals.

Background Jobs (via APScheduler):
  - daily_sync_job:  Runs at 08:30 AM IST (Mon–Fri) → downloads daily OHLCV,
                     updates InfluxDB + rebuilds parquet features + warms up Redis
  - live_sync_job:   Runs every 10 min, Mon–Fri 09:15–15:30 IST → scrapes live NSE
                     quotes, updates Redis micro-buffer, runs ML models,
                     caches top 20 BUY signals

REST Endpoints:
  GET  /health         → scheduler status + Redis / InfluxDB ping
  GET  /signals        → returns latest cached ML buy signals (top 20)
  POST /trigger-sync   → manually trigger a live_sync cycle immediately
  POST /trigger-daily  → manually trigger a full daily sync

Start the server:
    uvicorn live_trading_api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import logging
import warnings
import joblib
from contextlib import asynccontextmanager
from datetime import datetime, date

import pandas as pd
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from nse_scraper import NSEScraper
from state_manager_redis import MarketStateManager
from daily_sync import run_daily_sync
from seed_influx import InfluxWriter

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("LiveTradingAPI")

# ─────────────────────────────────────────────────────────────
# Paths & Config
# ─────────────────────────────────────────────────────────────
SAVED_DIR    = "saved_models"
FEATURES_DIR = "data/features"
TOP_N        = 20     # Number of top candidates to cache & return

# ─────────────────────────────────────────────────────────────
# Load ML Models at Startup (loaded once, reused every 10 min)
# ─────────────────────────────────────────────────────────────
logger.info("Loading pre-trained ML models...")
_required = ["lgbm.joblib", "xgb.joblib", "cat.joblib", "hgb.joblib",
             "imputer.joblib", "feature_cols.joblib", "ranked_feature_cols.joblib"]
_missing = [f for f in _required if not os.path.exists(os.path.join(SAVED_DIR, f))]
if _missing:
    raise FileNotFoundError(f"Missing model files: {_missing}. Run train_v2.py first.")

trained_lgbm        = joblib.load(f"{SAVED_DIR}/lgbm.joblib")
trained_xgb         = joblib.load(f"{SAVED_DIR}/xgb.joblib")
trained_cat         = joblib.load(f"{SAVED_DIR}/cat.joblib")
trained_hgb         = joblib.load(f"{SAVED_DIR}/hgb.joblib")
imputer             = joblib.load(f"{SAVED_DIR}/imputer.joblib")
feature_cols        = joblib.load(f"{SAVED_DIR}/feature_cols.joblib")
ranked_feature_cols = joblib.load(f"{SAVED_DIR}/ranked_feature_cols.joblib")
logger.info("Models loaded successfully.")

# Global shared objects
scraper = NSEScraper(request_delay=0.25)
manager = MarketStateManager()
writer = InfluxWriter()


# ─────────────────────────────────────────────────────────────
# Feature Engineering (inline, consistent with training)
# ─────────────────────────────────────────────────────────────

def _load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["52w_high"] = df["Close"].rolling(252).max()
    df["52w_low"]  = df["Close"].rolling(252).min()

    df["dist_SMA50"]      = (df["Close"] - df["SMA_50"])   / df["SMA_50"]
    df["dist_SMA150"]     = (df["Close"] - df["SMA_150"])  / df["SMA_150"]
    df["dist_SMA200"]     = (df["Close"] - df["SMA_200"])  / df["SMA_200"]
    df["dist_52w_high"]   = (df["Close"] - df["52w_high"]) / df["52w_high"]
    df["dist_52w_low"]    = (df["Close"] - df["52w_low"])  / df["52w_low"]
    df["SMA150_200_gap"]  = (df["SMA_150"] - df["SMA_200"]) / df["SMA_200"]
    df["SMA50_150_gap"]   = (df["SMA_50"]  - df["SMA_150"]) / df["SMA_150"]
    df["SMA200_slope_1m"] = (df["SMA_200"] - df["SMA_200"].shift(22))  / df["SMA_200"].shift(22)
    df["SMA200_slope_3m"] = (df["SMA_200"] - df["SMA_200"].shift(66))  / df["SMA_200"].shift(66)
    df["SMA200_slope_5m"] = (df["SMA_200"] - df["SMA_200"].shift(110)) / df["SMA_200"].shift(110)

    df["RSI_5d_change"]    = df["RSI"] - df["RSI"].shift(5)
    df["RSI_10d_change"]   = df["RSI"] - df["RSI"].shift(10)
    df["close_5d_return"]  = df["Close"].pct_change(5)
    df["close_10d_return"] = df["Close"].pct_change(10)
    df["close_20d_return"] = df["Close"].pct_change(20)
    df["volume_ratio"]     = df["Volume"].rolling(5).mean() / df["Volume"].rolling(20).mean()
    return df


# ─────────────────────────────────────────────────────────────
# Live Sync Job (Every 10 Minutes)
# ─────────────────────────────────────────────────────────────

def run_live_sync():
    """
    Core 10-minute pipeline:
    1. Scrape live NSE quotes for all 500 tickers
    2. Update Redis micro-buffer for each ticker
    3. Load parquet + inject live close → compute features
    4. Minervini filter → ML inference → rank top 20
    5. Cache top 20 signals in Redis
    """
    logger.info("─" * 50)
    logger.info("LIVE SYNC started")

    # Step 1: Get all tickers from features dir
    files = [f for f in os.listdir(FEATURES_DIR) if f.endswith(".parquet")]
    symbols = [f.replace(".parquet", "") for f in files]

    # Step 2: Scrape live quotes from NSE
    logger.info(f"Fetching live quotes for {len(symbols)} tickers...")
    live_quotes = scraper.fetch_all_quotes(symbols)
    logger.info(f"Got live quotes: {len(live_quotes)} tickers responded.")

    # Step 3: Update Redis micro-buffer and InfluxDB for each ticker that has a live quote
    today_date = pd.to_datetime(date.today())
    for symbol, quote in live_quotes.items():
        manager.update_live_from_nse(symbol, quote)

        # Upsert the live OHLCV for today in InfluxDB (using midnight timestamp replaces it in-place)
        df_live = pd.DataFrame([{
            "Open": quote["Open"],
            "High": quote["High"],
            "Low": quote["Low"],
            "Close": quote["Close"],
            "Volume": quote["Volume"]
        }], index=[today_date])
        writer.write_daily_ohlcv(symbol, df_live)

    # Step 4: Build cross-sectional snapshot for ML ranking
    snapshot_rows = []

    for fname in files:
        symbol = fname.replace(".parquet", "")
        try:
            df = _load_parquet(os.path.join(FEATURES_DIR, fname))
            df = df.sort_index()
            df = _add_features(df)

            # Use the live Close price if available; otherwise use latest parquet Close
            live_close = None
            if symbol in live_quotes:
                live_close = live_quotes[symbol]["Close"]

            latest = df.iloc[-1].copy()
            if live_close and live_close > 0:
                latest["Close"] = live_close   # Override with live price

            idx = len(df) - 1

            # Minervini trend conditions
            c1 = latest["Close"] > latest["SMA_150"] and latest["Close"] > latest["SMA_200"]
            c2 = latest["SMA_150"] > latest["SMA_200"]
            c3 = latest["SMA_200"] > df["SMA_200"].iloc[idx - 22] if idx >= 22 else False
            c4 = latest["SMA_50"] > latest["SMA_150"] and latest["SMA_50"] > latest["SMA_200"]

            last_252 = df["Close"].iloc[max(0, idx - 251):idx + 1]
            week52_high = last_252.max()
            week52_low  = last_252.min()
            c5 = latest["Close"] >= week52_low  * 1.30
            c6 = latest["Close"] <= week52_high * 1.25

            passed_macro = c1 and c2 and c3 and c4 and c5 and c6

            row = latest[feature_cols].to_dict()
            row["Symbol"]       = symbol
            row["Close"]        = latest["Close"]
            row["RSI"]          = latest["RSI"]
            row["Passed_Macro"] = passed_macro
            snapshot_rows.append(row)

        except Exception:
            pass

    if not snapshot_rows:
        logger.warning("No snapshot rows built. Skipping ML inference.")
        return

    snapshot_df = pd.DataFrame(snapshot_rows)

    # Cross-sectional percentile ranks
    for col in feature_cols:
        snapshot_df[f"{col}_rank"] = snapshot_df[col].rank(pct=True)

    # Filter by Minervini
    passed_df = snapshot_df[snapshot_df["Passed_Macro"]].reset_index(drop=True)
    logger.info(f"Scanned {len(snapshot_df)} | Passed Minervini: {len(passed_df)}")

    if passed_df.empty:
        manager.set_signals([])
        return

    # Step 5: ML Inference
    ranked_rows = []
    for _, row in passed_df.iterrows():
        symbol = row["Symbol"]
        try:
            feat = pd.DataFrame([row[ranked_feature_cols]], columns=ranked_feature_cols)
            feat_imputed = pd.DataFrame(imputer.transform(feat), columns=ranked_feature_cols)

            p_lgbm = float(trained_lgbm.predict_proba(feat_imputed)[0][1])
            p_xgb  = float(trained_xgb.predict_proba(feat_imputed)[0][1])
            p_cat  = float(trained_cat.predict_proba(feat_imputed)[0][1])
            p_hgb  = float(trained_hgb.predict_proba(feat_imputed)[0][1])
            final  = (p_lgbm + p_xgb + p_cat + p_hgb) / 4

            ranked_rows.append({
                "Symbol":      symbol,
                "Close":       round(float(row["Close"]), 2),
                "RSI":         round(float(row["RSI"]), 2),
                "LGBM_Score":  round(p_lgbm, 4),
                "XGB_Score":   round(p_xgb,  4),
                "CAT_Score":   round(p_cat,  4),
                "HGB_Score":   round(p_hgb,  4),
                "Final_Score": round(final,  4),
            })
        except Exception:
            pass

    ranked_df = pd.DataFrame(ranked_rows)
    if ranked_df.empty:
        manager.set_signals([])
        logger.info("No stocks ranked.")
        return

    ranked_df = ranked_df.sort_values("Final_Score", ascending=False).reset_index(drop=True)
    ranked_df["Rank"] = ranked_df.index + 1
    top_n = ranked_df.head(TOP_N).to_dict(orient="records")

    # Cache in Redis
    manager.set_signals(top_n)
    logger.info(f"Live sync complete — top {len(top_n)} signals cached in Redis.")


# ─────────────────────────────────────────────────────────────
# FastAPI App + Scheduler Lifecycle
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

    # Job 1: Daily sync at 8:30 AM IST, Mon–Fri
    scheduler.add_job(
        run_daily_sync,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=30, timezone="Asia/Kolkata"),
        id="daily_sync_job",
        replace_existing=True,
    )

    # Job 2: Live sync every 10 min during market hours Mon–Fri 9:15–15:30 IST
    scheduler.add_job(
        run_live_sync,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="15,25,35,45,55,5",   # 9:15, 9:25 … 15:25
            timezone="Asia/Kolkata",
        ),
        id="live_sync_job",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("APScheduler started — daily_sync @ 8:30 AM, live_sync every 10 min.")

    # Kick off an immediate live sync so the dashboard isn't empty on startup
    logger.info("Running initial live sync on startup...")
    try:
        run_live_sync()
    except Exception as e:
        logger.warning(f"Startup live sync failed: {e}")

    yield

    scheduler.shutdown(wait=False)
    logger.info("Scheduler shut down.")


app = FastAPI(
    title="StockMaster Live Trading API",
    description="Real-time NSE signal engine with Minervini + ML ensemble ranking",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Returns Redis connectivity and last signal cache timestamp."""
    redis_ok = manager.ping()
    cached   = manager.get_signals()
    return {
        "status":           "ok" if redis_ok else "redis_down",
        "redis":            "connected" if redis_ok else "disconnected",
        "signals_cached":   len(cached["signals"]),
        "signals_updated":  cached["updated_at"],
        "server_time_ist":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/signals")
def get_signals():
    """Returns the latest top-20 BUY signals from the Redis cache."""
    cached = manager.get_signals()
    if not cached["signals"]:
        return {"signals": [], "updated_at": cached["updated_at"], "message": "No signals yet. Market may be closed or sync pending."}
    return cached


@app.post("/trigger-sync")
def trigger_sync(background_tasks: BackgroundTasks):
    """Manually fire the live sync pipeline immediately."""
    background_tasks.add_task(run_live_sync)
    return {"message": "Live sync triggered in background."}


@app.post("/trigger-daily")
def trigger_daily(background_tasks: BackgroundTasks):
    """Manually fire the full daily data sync."""
    background_tasks.add_task(run_daily_sync)
    return {"message": "Daily sync triggered in background."}
