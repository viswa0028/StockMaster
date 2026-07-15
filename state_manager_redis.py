"""
state_manager_redis.py
----------------------
Manages all Redis state for StockMaster:
  - Macro context (200-day SMAs, 52w high/low) loaded from InfluxDB
  - Live 14-tick micro buffer updated every 10 minutes from NSE
  - ML prediction signals cache (top 20 BUY candidates)
"""

import redis
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Any
from influxdb_client import InfluxDBClient

logger = logging.getLogger("StateManagerRedis")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
REDIS_HOST = "localhost"
REDIS_PORT = 6379
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "my-super-secret-auth-token"
INFLUX_ORG = "stockmaster"
INFLUX_BUCKET = "market_data"

SIGNALS_CACHE_KEY = "live:signals"          # Stores latest ML top-20 as JSON
SIGNALS_TIMESTAMP_KEY = "live:signals:ts"   # Last update time


class MarketStateManager:
    def __init__(self):
        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.influx_url = INFLUX_URL
        self.influx_token = INFLUX_TOKEN
        self.influx_org = INFLUX_ORG
        self.bucket = INFLUX_BUCKET
        self.live_window_size = 14      # 14 ticks = 14 × 10 min for RSI-14

    # ─────────────────────────────────────────────────────────────
    # Macro Context (from InfluxDB → Redis)
    # ─────────────────────────────────────────────────────────────

    def warmup_daily_context(self):
        """
        Reads last 252 daily closes from InfluxDB for every ticker,
        calculates SMA-50/150/200 and 52-week high/low, stores in Redis.
        Should be called once at market open (08:30 IST weekdays).
        """
        logger.info("Starting daily macro context warmup from InfluxDB...")
        client = InfluxDBClient(url=self.influx_url, token=self.influx_token, org=self.influx_org)
        query_api = client.query_api()

        flux_query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: -2y)
              |> filter(fn: (r) => r._field == "Close")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
              |> group(columns: ["_measurement"])
              |> tail(n: 252)
        '''

        tables = query_api.query(flux_query)
        client.close()

        tickers_loaded = 0
        for table in tables:
            if not table.records:
                continue
            symbol = table.records[0].get_measurement()
            closes = [r.values.get("Close") for r in table.records if r.values.get("Close") is not None]

            if len(closes) < 200:
                continue

            df = pd.DataFrame({"Close": closes})
            context = {
                "SMA_20":          round(float(df["Close"].rolling(20).mean().iloc[-1]), 4),
                "SMA_50":          round(float(df["Close"].rolling(50).mean().iloc[-1]), 4),
                "SMA_150":         round(float(df["Close"].rolling(150).mean().iloc[-1]), 4),
                "SMA_200":         round(float(df["Close"].rolling(200).mean().iloc[-1]), 4),
                "SMA_200_prev_month": round(float(df["Close"].rolling(200).mean().iloc[-22]), 4),
                "Week_52_High":    round(float(df["Close"].max()), 4),
                "Week_52_Low":     round(float(df["Close"].min()), 4),
                "Last_Close":      round(float(df["Close"].iloc[-1]), 4),
            }
            redis_key = f"ticker:{symbol}:macro"
            self.r.hset(redis_key, mapping=context)
            tickers_loaded += 1

        logger.info(f"Macro warmup complete: {tickers_loaded} tickers loaded into Redis.")
        return tickers_loaded

    # ─────────────────────────────────────────────────────────────
    # Live Micro Buffer (NSE live quotes → Redis)
    # ─────────────────────────────────────────────────────────────

    def update_live_from_nse(self, symbol: str, quote_dict: Dict):
        """
        Called every 10 minutes with a fresh NSE quote.
        Pushes the new candle into the 14-tick rolling buffer in Redis.
        Also updates the live close price in the macro key.
        """
        redis_key = f"ticker:{symbol}:micro"
        self.r.rpush(redis_key, json.dumps(quote_dict))
        self.r.ltrim(redis_key, -self.live_window_size, -1)

        # Update live close in macro context so ML features use live price
        macro_key = f"ticker:{symbol}:macro"
        if self.r.exists(macro_key):
            self.r.hset(macro_key, "Live_Close", round(float(quote_dict.get("Close", 0)), 4))

    def update_live_minute(self, symbol: str, new_candle_dict: Dict):
        """Legacy alias for live_feeder.py compatibility."""
        self.update_live_from_nse(symbol, new_candle_dict)

    # ─────────────────────────────────────────────────────────────
    # Full State Retrieval (for ML inference)
    # ─────────────────────────────────────────────────────────────

    def get_trading_state(self, symbol: str) -> Optional[Dict]:
        """
        Returns combined macro (daily) + micro (live 14-tick) state for a ticker.
        Returns None if the ticker has no macro context (failed 200-day warmup).
        """
        macro_key = f"ticker:{symbol}:macro"
        micro_key = f"ticker:{symbol}:micro"

        macro_data = self.r.hgetall(macro_key)
        if not macro_data:
            return None

        macro_data = {k: float(v) for k, v in macro_data.items()}
        raw_micro = self.r.lrange(micro_key, 0, -1)
        micro_buffer = [json.loads(row) for row in raw_micro]

        return {
            "macro_context": macro_data,
            "micro_buffer":  micro_buffer,
        }

    # ─────────────────────────────────────────────────────────────
    # ML Signals Cache
    # ─────────────────────────────────────────────────────────────

    def set_signals(self, signals: List[Dict]):
        """
        Cache the latest ML top-N BUY predictions in Redis as a JSON string.
        Called at the end of every live_sync cycle.
        """
        self.r.set(SIGNALS_CACHE_KEY, json.dumps(signals))
        self.r.set(SIGNALS_TIMESTAMP_KEY, datetime.now().isoformat())
        logger.info(f"Cached {len(signals)} signals in Redis.")

    def get_signals(self) -> Dict:
        """
        Returns the last cached ML signals and the timestamp they were computed.
        """
        raw = self.r.get(SIGNALS_CACHE_KEY)
        ts  = self.r.get(SIGNALS_TIMESTAMP_KEY)
        return {
            "signals":    json.loads(raw) if raw else [],
            "updated_at": ts or "Never",
        }

    # ─────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        try:
            return self.r.ping()
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────
# Manual test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = MarketStateManager()
    print("Redis ping:", manager.ping())
    manager.warmup_daily_context()
    state = manager.get_trading_state("HDFCBANK.NS")
    if state:
        print("Macro:", state["macro_context"])
        print("Micro buffer length:", len(state["micro_buffer"]))
    else:
        print("No state found. Run warmup_daily_context first.")