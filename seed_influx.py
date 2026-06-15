"""
seed_influx.py
--------------
Reusable InfluxDB writer class for StockMaster.
Replaces the old one-shot seeding script.

Usage:
    from seed_influx import InfluxWriter
    writer = InfluxWriter()
    writer.write_daily_ohlcv("HDFCBANK.NS", df)   # df: DatetimeIndex, OHLCV columns
    writer.seed_all_from_parquet()                  # Seeds full features dir

Run directly to do a full initial seed from local parquet files:
    python seed_influx.py
"""

import os
import logging
import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

logger = logging.getLogger("InfluxWriter")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
INFLUX_URL   = "http://localhost:8086"
INFLUX_TOKEN = "my-super-secret-auth-token"
INFLUX_ORG   = "stockmaster"
INFLUX_BUCKET = "market_data"
FEATURES_DIR  = "data/features"


class InfluxWriter:
    """
    Handles all InfluxDB writes for StockMaster.
    Maintains a single synchronous write API connection.
    """

    def __init__(self):
        self.client = InfluxDBClient(
            url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def close(self):
        self.client.close()

    # ─────────────────────────────────────────────────────────────
    # Core Write
    # ─────────────────────────────────────────────────────────────

    def write_daily_ohlcv(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Writes a DataFrame of daily OHLCV data for `symbol` into InfluxDB.

        Args:
            symbol: Ticker string e.g. "HDFCBANK.NS"
            df:     DataFrame with DatetimeIndex and columns Open/High/Low/Close/Volume.
                    Timezone-naive indexes will be localized to UTC automatically.

        Returns True on success.
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol}, skipping.")
            return False

        df_write = df.copy()

        # Keep only the OHLCV columns that exist
        cols_to_keep = ["Open", "High", "Low", "Close", "Volume"]
        df_write = df_write[[c for c in cols_to_keep if c in df_write.columns]]

        # Handle MultiIndex columns from yfinance
        if isinstance(df_write.columns, pd.MultiIndex):
            df_write.columns = df_write.columns.get_level_values(0)

        # Ensure DatetimeIndex with UTC timezone (required by Influx)
        if not isinstance(df_write.index, pd.DatetimeIndex):
            logger.error(f"{symbol}: index is not DatetimeIndex.")
            return False

        if df_write.index.tz is None:
            df_write.index = df_write.index.tz_localize("UTC")

        try:
            self.write_api.write(
                bucket=INFLUX_BUCKET,
                record=df_write,
                data_frame_measurement_name=symbol,
            )
            logger.debug(f"Written {len(df_write)} records for {symbol}.")
            return True
        except Exception as e:
            logger.error(f"InfluxDB write failed for {symbol}: {e}")
            return False

    # ─────────────────────────────────────────────────────────────
    # Bulk Seed from Local Parquet Files
    # ─────────────────────────────────────────────────────────────

    def seed_all_from_parquet(self, features_dir: str = FEATURES_DIR):
        """
        Seeds InfluxDB with all local parquet feature files.
        Useful for the initial setup or a full reseed after data gaps.
        """
        files = [f for f in os.listdir(features_dir) if f.endswith(".parquet")]
        logger.info(f"Seeding {len(files)} tickers into InfluxDB...")

        success = 0
        for file in files:
            symbol = file.replace(".parquet", "")
            path = os.path.join(features_dir, file)
            try:
                df = pd.read_parquet(path)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.loc[:, ~df.columns.duplicated()]
                if self.write_daily_ohlcv(symbol, df):
                    success += 1
            except Exception as e:
                logger.error(f"Failed for {symbol}: {e}")

        logger.info(f"Seeding complete: {success}/{len(files)} tickers written.")
        return success

    # ─────────────────────────────────────────────────────────────
    # Append New Records (used by daily_sync.py)
    # ─────────────────────────────────────────────────────────────

    def append_new_records(self, symbol: str, new_df: pd.DataFrame) -> bool:
        """
        Appends only the newest rows to InfluxDB for a ticker.
        InfluxDB deduplicates by timestamp, so writing overlapping rows is safe.
        """
        return self.write_daily_ohlcv(symbol, new_df)


# ─────────────────────────────────────────────────────────────
# Run directly for a full initial seed
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    writer = InfluxWriter()
    try:
        writer.seed_all_from_parquet()
    finally:
        writer.close()