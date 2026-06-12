import redis
import json
import pandas as pd
from influxdb_client import InfluxDBClient

class MarketStateManager:
    def __init__(self):
        # Connect to your local Docker Redis
        self.r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # InfluxDB Configuration
        self.influx_url = "http://localhost:8086"
        self.influx_token = "my-super-secret-auth-token"
        self.influx_org = "stockmaster"
        self.bucket = "market_data"
        
        # We only need 14 minutes to calculate a live 14-period RSI
        self.live_window_size = 14 

    def warmup_daily_context(self):
        print("Calculating Daily Macro Context from InfluxDB...")
        client = InfluxDBClient(url=self.influx_url, token=self.influx_token, org=self.influx_org)
        query_api = client.query_api()

        # UPDATED FLUX QUERY: Grouping by _measurement
        flux_query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: -2y)
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
              |> group(columns: ["_measurement"])
              |> tail(n: 252)
        '''
        
        tables = query_api.query(flux_query)
        self.r.flushdb() 

        tickers_loaded = 0
        for table in tables:
            # Safely grab the measurement name (which is now our ticker!)
            symbol = table.records[0].get_measurement()
            
            closes = [r.values.get("Close") for r in table.records]
            
            if len(closes) < 200:
                continue 
            
            df = pd.DataFrame({'Close': closes})
            
            context = {
                "SMA_50": df['Close'].rolling(50).mean().iloc[-1],
                "SMA_150": df['Close'].rolling(150).mean().iloc[-1],
                "SMA_200": df['Close'].rolling(200).mean().iloc[-1],
                "SMA_200_prev_month": df['Close'].rolling(200).mean().iloc[-22],
                "Week_52_High": df['Close'].max(),
                "Week_52_Low": df['Close'].min()
            }
            
            redis_key = f"ticker:{symbol}:macro"
            self.r.hset(redis_key, mapping=context)
            tickers_loaded += 1

        print(f"Warm-Up Complete! {tickers_loaded} tickers have macro context locked in.")

    def update_live_minute(self, symbol, new_candle_dict):
        """
        Runs EVERY MINUTE during live trading. 
        Pushes the new candle and trims the list to exactly 14 items.
        """
        redis_key = f"ticker:{symbol}:micro"
        
        self.r.rpush(redis_key, json.dumps(new_candle_dict))
        # Keep only the last 14 minutes
        self.r.ltrim(redis_key, -self.live_window_size, -1)

    def get_trading_state(self, symbol):
        """
        The bridge for your ML Model and Rules Engine.
        Returns both the static daily data and the rolling 14-min buffer.
        """
        macro_key = f"ticker:{symbol}:macro"
        micro_key = f"ticker:{symbol}:micro"
        
        # 1. Fetch the daily static data
        macro_data = self.r.hgetall(macro_key)
        if not macro_data:
            return None # Stock didn't pass the 200-day warmup
            
        # Redis returns strings, convert them back to floats
        macro_data = {k: float(v) for k, v in macro_data.items()}
        
        # 2. Fetch the live 14-minute buffer
        raw_micro = self.r.lrange(micro_key, 0, -1)
        micro_buffer = [json.loads(row) for row in raw_micro]
        
        return {
            "macro_context": macro_data,
            "micro_buffer": micro_buffer
        }

# --- To Test It ---
if __name__ == "__main__":
    manager = MarketStateManager()
    
    # 1. Run the warmup
    manager.warmup_daily_context()
    
    # 2. Test pulling the full state for a stock
    state = manager.get_trading_state("AAPL") # Use a valid ticker
    
    if state:
        print("\nSuccessfully retrieved Multi-Timeframe State:")
        print("Macro:", state["macro_context"])
        print("Micro Buffer Length:", len(state["micro_buffer"]))