import yfinance as yf
import time
from datetime import datetime
from state_manager_redis import MarketStateManager

def run_live_feeder(tickers):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Live Data Feeder for {len(tickers)} tickers...")
    manager = MarketStateManager()
    
    while True:
        try:
            # Download the latest 1-minute data for all tickers
            # group_by='ticker' helps us parse multiple stocks easily
            data = yf.download(tickers, period="1d", interval="1m", group_by='ticker', progress=False)
            
            for symbol in tickers:
                # Handle single vs multiple ticker format from yfinance
                df = data[symbol] if len(tickers) > 1 else data
                
                if df.empty:
                    continue
                    
                # Grab the absolute latest 1-minute candle
                latest = df.iloc[-1]
                
                candle_dict = {
                    "time": latest.name.isoformat(),
                    "Open": float(latest['Open']),
                    "High": float(latest['High']),
                    "Low": float(latest['Low']),
                    "Close": float(latest['Close']),
                    "Volume": int(latest['Volume'])
                }
                
                # Push it into the Redis 14-minute rolling buffer
                manager.update_live_minute(symbol, candle_dict)
                
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully pushed live minute to Redis.")
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error fetching data: {e}")
            
        # Wait 60 seconds before fetching the next live candle
        time.sleep(60)

if __name__ == "__main__":
    # Ensure these exactly match the tickers you warmed up in Redis
    active_tickers = ["PFIZER.NS", "CUMMINSIND.NS", "BRIGADE.NS", "HDFCBANK.NS", "COFORGE.NS"]
    run_live_feeder(active_tickers)