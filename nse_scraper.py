"""
nse_scraper.py
--------------
Scrapes live OHLCV quotes from NSE India's internal API.
NSE requires a valid browser session (cookies) before answering data requests.
This module handles:
  1. Cookie acquisition by visiting the NSE homepage first
  2. Automatic session refresh when cookies expire (HTTP 401)
  3. Bulk quote fetching for all 500 tickers with graceful error handling
"""

import requests
import time
import logging
from datetime import datetime
from typing import Optional, Dict, List

logger = logging.getLogger("NSEScraper")

# NSE endpoints
NSE_HOME_URL = "https://www.nseindia.com"
NSE_QUOTE_URL = "https://www.nseindia.com/api/quote-equity"
NSE_MARKET_STATUS_URL = "https://www.nseindia.com/api/marketStatus"

# Headers that mimic a real Chrome browser — required by NSE/Akamai
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
    "X-Requested-With": "XMLHttpRequest",
}


class NSEScraper:
    """
    Manages an NSE browser-like session and provides quote-fetching methods.
    Thread-safe enough for a single-threaded scheduler.
    """

    def __init__(self, request_delay: float = 0.3):
        """
        Args:
            request_delay: Seconds to wait between individual ticker requests
                           to avoid triggering NSE rate limits. Default 0.3s.
        """
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self.request_delay = request_delay
        self._cookies_valid = False
        self._last_cookie_refresh: Optional[datetime] = None

    # ─────────────────────────────────────────────────────────────
    # Session / Cookie Management
    # ─────────────────────────────────────────────────────────────

    def refresh_session(self) -> bool:
        """
        Visit NSE homepage to acquire fresh cookies (nsit, nseappid, etc.).
        Must be called before any data API request.
        Returns True on success.
        """
        try:
            resp = self.session.get(NSE_HOME_URL, timeout=10)
            if resp.status_code == 200:
                self._cookies_valid = True
                self._last_cookie_refresh = datetime.now()
                logger.info("NSE session refreshed successfully.")
                return True
            else:
                logger.warning(f"NSE homepage returned {resp.status_code}")
                self._cookies_valid = False
                return False
        except requests.RequestException as e:
            logger.error(f"Failed to refresh NSE session: {e}")
            self._cookies_valid = False
            return False

    def _ensure_session(self):
        """Auto-refresh session if cookies are stale or missing."""
        if not self._cookies_valid:
            self.refresh_session()

    # ─────────────────────────────────────────────────────────────
    # Market Status
    # ─────────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        """Returns True if NSE equity market is currently open."""
        self._ensure_session()
        try:
            resp = self.session.get(NSE_MARKET_STATUS_URL, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                for market in data.get("marketState", []):
                    if market.get("market") == "Capital Market":
                        return market.get("marketStatus", "").lower() == "open"
        except Exception as e:
            logger.warning(f"Could not check market status: {e}")
        return False

    # ─────────────────────────────────────────────────────────────
    # Single Quote Fetch
    # ─────────────────────────────────────────────────────────────

    def fetch_quote(self, symbol: str) -> Optional[Dict]:
        """
        Fetch a single live OHLCV quote from NSE for `symbol`.
        `symbol` should be the raw NSE symbol without '.NS' (e.g., 'RELIANCE').

        Returns a dict with keys: symbol, Open, High, Low, Close, Volume, timestamp
        or None on failure.
        """
        self._ensure_session()
        params = {"symbol": symbol.replace(".NS", "").upper()}
        try:
            resp = self.session.get(NSE_QUOTE_URL, params=params, timeout=8)

            # Session expired — refresh and retry once
            if resp.status_code in (401, 403):
                logger.info(f"Session expired for {symbol}. Refreshing...")
                self._cookies_valid = False
                self._ensure_session()
                resp = self.session.get(NSE_QUOTE_URL, params=params, timeout=8)

            if resp.status_code != 200:
                logger.debug(f"Non-200 for {symbol}: {resp.status_code}")
                return None

            data = resp.json()
            price_info = data.get("priceInfo", {})
            intradata = data.get("metadata", {})

            close = float(price_info.get("lastPrice", 0) or 0)
            open_ = float(price_info.get("open", close) or close)
            high = float(price_info.get("intraDayHighLow", {}).get("max", close) or close)
            low = float(price_info.get("intraDayHighLow", {}).get("min", close) or close)
            volume = int(intradata.get("totalTradedVolume", 0) or 0)

            if close == 0:
                return None

            return {
                "symbol": symbol.replace(".NS", "") + ".NS",
                "Open": round(open_, 2),
                "High": round(high, 2),
                "Low": round(low, 2),
                "Close": round(close, 2),
                "Volume": volume,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.debug(f"Error fetching {symbol}: {e}")
            return None

    # ─────────────────────────────────────────────────────────────
    # Bulk Quote Fetch
    # ─────────────────────────────────────────────────────────────

    def fetch_all_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch quotes for a list of symbols. Respects `request_delay` between
        each call so NSE doesn't rate-limit the session.

        Returns a dict keyed by symbol (with .NS suffix) containing the quote dict.
        """
        self._ensure_session()
        results = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            quote = self.fetch_quote(symbol)
            if quote:
                results[quote["symbol"]] = quote

            # Small delay to stay under NSE rate limits
            if self.request_delay > 0:
                time.sleep(self.request_delay)

            if (i + 1) % 50 == 0:
                logger.info(f"Fetched {i+1}/{total} quotes...")

        logger.info(f"Bulk fetch complete: {len(results)}/{total} succeeded.")
        return results


# ─────────────────────────────────────────────────────────────
# Quick sanity-check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = NSEScraper(request_delay=0.5)
    scraper.refresh_session()
    print("Market open?", scraper.is_market_open())
    q = scraper.fetch_quote("HDFCBANK")
    print("HDFCBANK quote:", q)
