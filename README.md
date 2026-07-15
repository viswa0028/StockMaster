# StockMaster v2

StockMaster v2 is a real-time, event-driven algorithmic trading system built for the Indian stock market (NSE Nifty 500). It transitions from a batch End-of-Day (EOD) system to a live infrastructure using a completely local, zero-cost stack. 

The system continuously screens stocks using the Minervini Trend Template and ranks them using a cross-sectional ensemble of machine learning models (LightGBM, XGBoost, CatBoost, HistGradientBoosting) to provide real-time BUY signals.

## System Architecture

To keep hosting costs at zero, the entire stack is designed to run locally using Docker and open-source tools.

1. **Data Layer**: 
    - End-of-day data is fetched from `yfinance`.
    - Live intraday quotes are scraped directly from the NSE API.
2. **Storage Layer**: 
    - **InfluxDB** (Local): Time-series database storing historical daily bars (OHLCV).
    - **Redis** (Local): Fast in-memory state manager storing rolling 14-minute intraday micro-buffers and daily macro contexts.
3. **Processing Layer**: 
    - Computes real-time technical features (RSI, SMAs).
    - Checks Minervini macro-trend rules.
    - Generates cross-sectional percentile rankings.
4. **Execution Layer**:
    - FastAPI background scheduler evaluates stocks every 10 minutes.
    - Streamlit dashboard displays the top-ranked BUY signals.

## Setup & Installation

### 1. Prerequisites
- **Python 3.9+**
- **Docker & Docker Compose** (for InfluxDB and Redis)

### 2. Start Databases
Run the provided Docker Compose file to spin up InfluxDB and Redis:
```bash
docker-compose up -d
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: Ensure you have pandas, numpy, lightgbm, xgboost, catboost, scikit-learn, fastapi, uvicorn, streamlit, influxdb-client, redis, and yfinance installed).*

## Key Components

### 🔄 Data & State Management
- `data_nse.py`: Downloads historical Nifty 500 data from yfinance.
- `nse_scraper.py`: Handles live quote scraping directly from NSE with session management.
- `daily_sync.py`: Daily pipeline that fetches missing data, updates parquets, writes to InfluxDB, and warms up the Redis cache.
- `state_manager_redis.py`: Controls interactions with Redis and InfluxDB for storing macro contexts and live micro-buffers.

### 🧠 Machine Learning & Inference
- `preprocessing.py`: Feature engineering script calculating moving averages, RSI, and volatility.
- `train_v2.py`: Generates cross-sectional rank targets, performs walk-forward validation, and trains the ensemble of tree-based models. Saves the models to `saved_models/`.
- `inference_v2.py`: Offline inference script that processes daily snapshots and ranks candidates using the pre-trained models.

### 🚀 APIs & UI
- `live_trading_api.py`: FastAPI server that runs scheduled jobs (`live_sync` every 10 mins during market hours, and `daily_sync`). Serves endpoints for health and live signals.
- `streamlit_app.py`: Real-time dashboard that pulls data from the FastAPI backend to visualize the top BUY signals.

### 🧪 Backtesting
- `market_reply_engine_testing_v1.py`: A historical playback engine to validate strategies on past trading days.

## Usage Guide

**1. Initial Setup (One-time):**
Download the initial batch of historical data and run the preprocessing scripts to build the feature parquets. 
```bash
python data_nse.py
python preprocessing.py
```

**2. Train Models:**
Train the LightGBM, XGBoost, CatBoost, and HistGradientBoosting ensemble.
```bash
python train_v2.py
```

**3. Start the Live API Server:**
This kicks off the APScheduler (Daily sync at 8:30 AM, Live sync every 10 mins).
```bash
uvicorn live_trading_api:app --host 0.0.0.0 --port 8000
```

**4. Start the Dashboard:**
Open a new terminal and launch the Streamlit app.
```bash
streamlit run streamlit_app.py
```
