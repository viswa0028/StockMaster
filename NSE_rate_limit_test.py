"""
nse_rate_limit_test.py
-----------------------
One-hour smoke test for scraping NSE quote data. Run this FIRST, before
committing to a 2-3 week unattended collection, to check:

  1. Does NSE respond cleanly (200 + valid JSON) from wherever you run this?
  2. Does it start blocking/throttling after N requests or N minutes?
  3. What does a block actually look like (403? 401? empty body? captcha HTML?)
  4. What's typical latency, and does it degrade over the hour?

Run it from your cloud VM (or your laptop, to compare) for exactly one hour
during market hours. Everything gets logged to nse_rate_limit_test_log.csv
so you can inspect it afterward instead of relying on scrollback.

Usage:
    python nse_rate_limit_test.py
"""

import csv
import os
import time
from datetime import datetime

import requests

BASE_URL = "https://www.nseindia.com"
QUOTE_URL = "https://www.nseindia.com/api/quote-equity"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# A small, representative basket — not the full 500. The point of today's
# test is to characterize rate limits, not to collect real data yet.
TEST_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "ITC", "LT",
    "AXISBANK", "KOTAKBANK", "BHARTIARTL", "HINDUNILVR", "MARUTI",
    "SUNPHARMA", "TATAMOTORS", "WIPRO", "ADANIENT", "ONGC", "NTPC", "POWERGRID",
]

LOG_FILE = "nse_rate_limit_test_log.csv"
SECONDS_BETWEEN_SYMBOLS = 0.5   # gentle pacing within a cycle
CYCLE_INTERVAL_SECONDS = 60     # one full pass over TEST_SYMBOLS per minute
TEST_DURATION_SECONDS = 60 * 60 # 1 hour


def new_session():
    """NSE requires a warm-up hit to the homepage to set cookies before
    the API endpoints will respond with real data instead of a block page."""
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get(BASE_URL, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"  [warn] homepage warm-up failed: {e}")
    return s


def fetch_quote(session, symbol):
    t0 = time.time()
    try:
        resp = session.get(QUOTE_URL, params={"symbol": symbol}, timeout=10)
        latency = time.time() - t0
        status = resp.status_code
        blocked_signal = False
        body_snippet = ""
        ok = False

        if status == 200:
            try:
                data = resp.json()
                price = data.get("priceInfo", {}).get("lastPrice")
                ok = price is not None
                if not ok:
                    body_snippet = str(data)[:200]
            except ValueError:
                # 200 but not JSON — usually a captcha/HTML block page
                blocked_signal = True
                body_snippet = resp.text[:200]
        else:
            if status in (401, 403, 429):
                blocked_signal = True
            body_snippet = resp.text[:200]

        return {
            "status_code": status,
            "latency_sec": round(latency, 3),
            "ok": ok,
            "blocked_signal": blocked_signal,
            "body_snippet": body_snippet.replace("\n", " "),
        }
    except requests.exceptions.RequestException as e:
        return {
            "status_code": None,
            "latency_sec": round(time.time() - t0, 3),
            "ok": False,
            "blocked_signal": False,
            "body_snippet": str(e)[:200],
        }


def main():
    print(f"Starting 1-hour NSE rate-limit test — logging to {LOG_FILE}")
    session = new_session()
    start = time.time()
    cycle = 0
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "cycle", "symbol", "status_code",
                "latency_sec", "ok", "blocked_signal", "body_snippet",
            ])

        while time.time() - start < TEST_DURATION_SECONDS:
            cycle += 1
            cycle_start = time.time()
            print(f"\n--- Cycle {cycle} @ {datetime.now().strftime('%H:%M:%S')} ---")
            consecutive_failures = 0

            for symbol in TEST_SYMBOLS:
                result = fetch_quote(session, symbol)
                ts = datetime.now().isoformat()
                writer.writerow([
                    ts, cycle, symbol, result["status_code"],
                    result["latency_sec"], result["ok"],
                    result["blocked_signal"], result["body_snippet"],
                ])
                f.flush()

                label = "OK" if result["ok"] else ("BLOCKED?" if result["blocked_signal"] else "FAIL")
                print(f"  {symbol:12} {label:9} status={result['status_code']} "
                      f"latency={result['latency_sec']}s")

                if not result["ok"]:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        print("  5 consecutive failures — refreshing session (new cookies)...")
                        session = new_session()
                        consecutive_failures = 0
                else:
                    consecutive_failures = 0

                time.sleep(SECONDS_BETWEEN_SYMBOLS)

            elapsed_cycle = time.time() - cycle_start
            sleep_time = max(0, CYCLE_INTERVAL_SECONDS - elapsed_cycle)
            print(f"Cycle took {elapsed_cycle:.1f}s, sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)

    print(f"\nDone. {cycle} cycles completed. Review {LOG_FILE} before deciding on the full run.")


if __name__ == "__main__":
    main()

