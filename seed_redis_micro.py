import redis
import json
from datetime import datetime, timedelta

def seed_dummy_micro_data(symbol="PFIZER.NS"):
    # Connect to local Docker Redis
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_key = f"ticker:{symbol}:micro"
    
    # Clear any existing micro buffer for this clean test
    r.delete(redis_key)
    
    # Base price to start simulating from
    base_price = 150.0
    start_time = datetime.now() - timedelta(minutes=14)
    
    # 14 simulated 1-minute candles with slight random variations
    # Making the close prices generally increase to simulate an upward RSI momentum
    price_deltas = [0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.3, -0.1, 0.4, 0.6, 0.2, -0.3, 0.5, 0.4]
    
    candles = []
    current_close = base_price
    
    for i in range(14):
        candle_time = start_time + timedelta(minutes=i)
        
        open_price = current_close
        close_price = open_price + price_deltas[i]
        high_price = max(open_price, close_price) + 0.15
        low_price = min(open_price, close_price) - 0.10
        volume = 5000 + (i * 200) # Incremental volume
        
        candle_dict = {
            "time": candle_time.isoformat(),
            "Open": round(open_price, 2),
            "High": round(high_price, 2),
            "Low": round(low_price, 2),
            "Close": round(close_price, 2),
            "Volume": int(volume)
        }
        
        candles.append(json.dumps(candle_dict))
        # Set next open to current close
        current_close = close_price

    # Bulk push all 14 candles to the right side of the list
    r.rpush(redis_key, *candles)
    
    print(f"Successfully injected {r.llen(redis_key)} dummy candles into Redis for {symbol}!")
    print("Sample of the latest candle injected:")
    print(json.loads(candles[-1]))

if __name__ == "__main__":
    seed_dummy_micro_data("PFIZER.NS")