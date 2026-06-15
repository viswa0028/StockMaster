"""
daily_sync.py
-------------
Handles the once-per-day data pipeline:

  1. Detect the last checkpoint date from local parquet files
  2. Download missing OHLCV data from NSE (via NSE CSV archives or yfinance fallback)
  3. Rebuild technical features (SMA, RSI) in parquet files
  4. Write new daily bars to InfluxDB
  5. Trigger Redis macro warmup for next trading session

Run directly to do a manual daily sync:
    python daily_sync.py

Or import and call:
    from daily_sync import run_daily_sync
    run_daily_sync()
"""

import os
import logging
import time
import warnings
from datetime import datetime, timedelta, date

import pandas as pd
import yfinance as yf

from seed_influx import InfluxWriter
from state_manager_redis import MarketStateManager

warnings.filterwarnings("ignore")
logger = logging.getLogger("DailySync")

FEATURES_DIR = "data/features"
RAW_DIR = "data/raw"


# ─────────────────────────────────────────────────────────────
# Feature Engineering (same as preprocessing.py)
# ─────────────────────────────────────────────────────────────

def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SMA and RSI features on a raw OHLCV DataFrame."""
    df = df.copy()

    # Handle MultiIndex from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    df.dropna(subset=["Close"], inplace=True)

    # Moving Averages
    df["SMA_10"]  = df["Close"].rolling(10).mean()
    df["SMA_20"]  = df["Close"].rolling(20).mean()
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_150"] = df["Close"].rolling(150).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    # Price features
    df["daily_range"]    = df["High"] - df["Low"]
    df["open_close_gap"] = df["Close"] - df["Open"]
    df["high_close_gap"] = df["High"] - df["Close"]
    df["low_close_gap"]  = df["Close"] - df["Low"]
    df["gap_up_down"]    = df["Open"] - df["Close"].shift(1)

    # RSI 14
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # RSI signal
    df["RSI_signal"] = 0
    df.loc[df["RSI"] > 55, "RSI_signal"] = 1
    df.loc[df["RSI"] < 45, "RSI_signal"] = -1

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# Checkpoint Detection
# ─────────────────────────────────────────────────────────────

def get_last_checkpoint() -> tuple[pd.Timestamp | None, list[str]]:
    """
    Reads the features dir, finds the most recent date across all parquet files,
    and returns (last_date, list_of_parquet_filenames).
    """
    if not os.path.exists(FEATURES_DIR):
        return None, []

    files = [f for f in os.listdir(FEATURES_DIR) if f.endswith(".parquet")]
    last_date = None

    for fname in files[:20]:   # Sample first 20 to find the checkpoint fast
        try:
            df = pd.read_parquet(os.path.join(FEATURES_DIR, fname))
            if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
                idx = df.index.tz_localize(None) if df.index.tz else df.index
                candidate = idx.max()
                if last_date is None or candidate > last_date:
                    last_date = candidate
        except Exception:
            continue

    return last_date, files


# ─────────────────────────────────────────────────────────────
# Per-Ticker Download + Update
# ─────────────────────────────────────────────────────────────

def _download_missing(symbol: str, start: date, end: date) -> pd.DataFrame | None:
    """
    Downloads OHLCV data for `symbol` from yfinance for [start, end].
    Returns a clean DataFrame or None on failure.

    Note: yfinance is used here for *daily* historical bars (end-of-day),
    NOT for live intraday data. Daily data from yfinance is generally reliable.
    """
    try:
        ticker_sym = symbol if symbol.endswith(".NS") else symbol + ".NS"
        df = yf.download(
            ticker_sym,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Close"])
    except Exception as e:
        logger.debug(f"yfinance download failed for {symbol}: {e}")
        return None


def _update_single_ticker(symbol: str, start: date, end: date, writer: InfluxWriter) -> bool:
    """
    Downloads new daily bars for one ticker, appends to parquet, writes to InfluxDB.
    Returns True if any new data was added.
    """
    file_name = symbol if symbol.endswith(".NS") else symbol + ".NS"
    parquet_path = os.path.join(FEATURES_DIR, f"{file_name}.parquet")

    new_df = _download_missing(file_name, start, end)
    if new_df is None or new_df.empty:
        return False

    # Merge with existing parquet data
    if os.path.exists(parquet_path):
        try:
            existing = pd.read_parquet(parquet_path)
            if isinstance(existing.columns, pd.MultiIndex):
                existing.columns = existing.columns.get_level_values(0)

            # Align timezone handling
            if existing.index.tz is not None and new_df.index.tz is None:
                new_df.index = new_df.index.tz_localize("UTC")
            elif existing.index.tz is None and new_df.index.tz is not None:
                new_df.index = new_df.index.tz_localize(None)

            combined_raw = pd.concat([existing, new_df])
            combined_raw = combined_raw[~combined_raw.index.duplicated(keep="last")]
            combined_raw.sort_index(inplace=True)
        except Exception as e:
            logger.warning(f"Could not merge parquet for {file_name}: {e}. Using new data only.")
            combined_raw = new_df
    else:
        combined_raw = new_df

    # Recompute features on the full dataset
    try:
        featured_df = _compute_features(combined_raw)
        featured_df.to_parquet(parquet_path)
    except Exception as e:
        logger.warning(f"Feature computation failed for {file_name}: {e}")
        return False

    # Write the new rows to InfluxDB
    writer.append_new_records(file_name, new_df)
    return True


# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────

def run_daily_sync(target_date: date | None = None):
    """
    Main daily sync function. Detects the data gap, downloads missing data
    for all 500 tickers, updates parquet + InfluxDB, then warms up Redis.

    Args:
        target_date: The most recent date to sync up to. Defaults to today.
    """
    if target_date is None:
        target_date = date.today()

    logger.info("=" * 60)
    logger.info(f"DAILY SYNC started — syncing up to {target_date}")
    logger.info("=" * 60)

    last_checkpoint, files = get_last_checkpoint()

    if last_checkpoint is None:
        logger.warning("No local data found. Please run data_nse.py and preprocessing.py first.")
        return

    last_checkpoint_date = last_checkpoint.date()
    logger.info(f"Last checkpoint: {last_checkpoint_date} | Target: {target_date}")

    if target_date <= last_checkpoint_date:
        logger.info("Data is already up to date. No sync needed.")
        return

    start_date = last_checkpoint_date + timedelta(days=1)
    logger.info(f"Downloading data from {start_date} to {target_date} for {len(files)} tickers...")

    writer = InfluxWriter()
    updated_count = 0

    for i, fname in enumerate(files):
        symbol = fname.replace(".parquet", "")
        try:
            if _update_single_ticker(symbol, start_date, target_date, writer):
                updated_count += 1
        except Exception as e:
            logger.warning(f"Skipped {symbol}: {e}")

        # Brief pause every 50 tickers to avoid API throttling
        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i+1}/{len(files)}")
            time.sleep(1)

    writer.close()
    logger.info(f"Daily sync complete: {updated_count}/{len(files)} tickers updated.")

    # Refresh Redis macro context with the new data
    logger.info("Refreshing Redis macro context...")
    try:
        manager = MarketStateManager()
        loaded = manager.warmup_daily_context()
        logger.info(f"Redis warmup done: {loaded} tickers refreshed.")
    except Exception as e:
        logger.error(f"Redis warmup failed: {e}")


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="StockMaster Daily Data Sync")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    run_daily_sync(target)
