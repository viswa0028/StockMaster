import pandas as pd
import numpy as np
import joblib
import json
from state_manager_redis import MarketStateManager # Your script from Phase 2

class LocalTradingSimulator:
    def __init__(self, model_dir="saved_models"):
        print("Loading trained models into RAM...")
        # Load the models you trained in Phase 1
        self.models = {
            'lgbm': joblib.load(f"{model_dir}/lgbm.joblib"),
            'xgb': joblib.load(f"{model_dir}/xgb.joblib"),
            'cat': joblib.load(f"{model_dir}/cat.joblib"),
            'hgb': joblib.load(f"{model_dir}/hgb.joblib")
        }
        self.imputer = joblib.load(f"{model_dir}/imputer.joblib")
        self.feature_cols = joblib.load(f"{model_dir}/ranked_feature_cols.joblib")
        
        # Connect to your Redis state manager
        self.state = MarketStateManager()

    def calculate_live_rsi(self, micro_buffer):
        """Calculates a live 14-period RSI from the Redis buffer"""
        if len(micro_buffer) < 14:
            return 50 # Default neutral if not enough data
            
        df = pd.DataFrame(micro_buffer)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def evaluate_stock(self, symbol):
        """Pulls the state, checks Minervini rules, and gets ML Confidence"""
        
        # 1. Fetch data from Redis
        market_state = self.state.get_trading_state(symbol)
        if not market_state:
            return None
            
        macro = market_state["macro_context"]
        micro = market_state["micro_buffer"]
        
        # Current live price is the last item in the micro buffer
        live_close = micro[-1]["Close"]
        
        # 2. Check the Minervini Macro Rules (The Bouncer)
        c1 = live_close > macro["SMA_150"] and live_close > macro["SMA_200"]
        c2 = macro["SMA_150"] > macro["SMA_200"]
        c3 = macro["SMA_200"] > macro["SMA_200_prev_month"]
        c4 = macro["SMA_50"] > macro["SMA_150"] and macro["SMA_50"] > macro["SMA_200"]
        c5 = live_close >= macro["Week_52_Low"] * 1.30
        c6 = live_close <= macro["Week_52_High"] * 1.25
        
        passed_macro = c1 and c2 and c3 and c4 and c5 and c6
        
        # if not passed_macro:
        #     return {"Symbol": symbol, "Status": "REJECTED (Failed Macro Trend)"}

        # 3. Calculate Live Intraday Features
        live_rsi = self.calculate_live_rsi(micro)
        
        # 4. Prepare data for the ML Model
        # In a real live system, you would calculate these for all passed stocks 
        # simultaneously to rank them cross-sectionally. For this single-stock test, 
        # we will simulate standard inputs.
        
        live_features = pd.DataFrame([{
            'RSI_rank': 0.85, # Simulated high rank
            'dist_SMA50_rank': 0.90,
            'dist_SMA150_rank': 0.80,
            'dist_SMA200_rank': 0.75,
            # ... (you would populate all required feature_cols here)
        }])
        
        # Fill in missing columns with 0.5 (median rank) just for this local test to run
        for col in self.feature_cols:
            if col not in live_features.columns:
                live_features[col] = 0.5
                
        # Reorder to match training exactly
        X_live = live_features[self.feature_cols]
        X_imputed = self.imputer.transform(X_live)
        X_imputed_df = pd.DataFrame(X_imputed, columns=self.feature_cols)
        # 5. Get the ML Confidence Score
        p_lgbm = self.models['lgbm'].predict_proba(X_imputed_df)[:, 1][0]
        p_xgb  = self.models['xgb'].predict_proba(X_imputed_df)[:, 1][0]
        p_cat  = self.models['cat'].predict_proba(X_imputed_df)[:, 1][0]
        p_hgb  = self.models['hgb'].predict_proba(X_imputed_df)[:, 1][0]
        
        # Average the probabilities
        ensemble_confidence = (p_lgbm + p_xgb + p_cat + p_hgb) / 4
        
        # 6. Final Execution Decision
        action = "HOLD"
        if ensemble_confidence >= 0.75 and live_rsi > 55:
            action = "STRONG BUY"
            
        return {
            "Symbol": symbol,
            "Live_Price": live_close,
            "Live_RSI": round(live_rsi, 2),
            "ML_Confidence": f"{ensemble_confidence * 100:.2f}%",
            "Action": action
        }

if __name__ == "__main__":
    sim = LocalTradingSimulator()
    # The list of tickers you want to monitor live
    test_symbols = ["PFIZER.NS", "CUMMINSIND.NS", "BRIGADE.NS", "HDFCBANK.NS", "COFORGE.NS"] 
    
    print("\n🚀 Starting Real-Time Live ML Scanner...")
    
    while True:
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f"\n--- ⏱️ Live Market Scan @ {current_time} ---")
        
        for sym in test_symbols:
            result = sim.evaluate_stock(sym)
            
            if result:
                # 1. Check if the stock is simply waiting for the live feeder to fill Redis
                if "AWAITING DATA" in result.get("Status", ""):
                    print(f"[{sym}] ⏳ Awaiting Data (Redis buffer empty)")
                
                # 2. Check if it was filtered out by the Minervini macro trend rules
                elif result.get("Status") == "REJECTED (Failed Macro Trend)":
                    print(f"[{sym}] 🔴 Rejected by Macro Trend")
                
                # 3. Valid ML evaluation output
                elif "ML_Confidence" in result:
                    print(f"[{sym}] 🟢 ML Score: {result['ML_Confidence']} | RSI: {result['Live_RSI']} | Action: {result['Action']}")
                    
        # Sleep for 60 seconds before pulling from Redis again
        time.sleep(60)


# testing redis

# import redis

# r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# # Find all macro keys currently in Redis
# macro_keys = r.keys("ticker:*:macro")

# print(f"Total stocks warmed up in Redis: {len(macro_keys)}")

# if len(macro_keys) > 0:
#     print("Here are the first 5 available tickers:")
#     for key in macro_keys[:5]:
#         print(f" - {key.split(':')[1]}")
# else:
#     print("Redis is empty! You need to run the warmup script.")