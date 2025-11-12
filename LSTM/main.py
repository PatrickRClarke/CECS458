# 10/4/'25

import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# --- Settings ---
symbol = "BTCUSDT"
interval = "5m"
weeks = 12.8571 #( 12.8571: 90 days)   
limit = 1000    # Binance's per-request max

# Binance US base URL
BASE_URL = "https://api.binance.us/api/v3/klines"

# --- Calculate time range ---
end_time = int(time.time() * 1000)  # current time in ms
start_time = int((datetime.utcnow() - timedelta(weeks=weeks)).timestamp() * 1000)

print(f"Fetching {symbol} {interval} data from {datetime.utcfromtimestamp(start_time/1000)} "
      f"to {datetime.utcfromtimestamp(end_time/1000)}")

# --- Fetch loop ---
all_klines = []
while True:
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "startTime": start_time,
        "endTime": end_time
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if not data:
        break

    all_klines.extend(data)

    # Get the close time of the last kline and add 1 ms for next batch
    last_close_time = data[-1][6]
    start_time = last_close_time + 1

    # Binance enforces a tiny rate limit; short sleep keeps things polite
    time.sleep(0.1)

    # Stop if we already reached end_time
    if last_close_time >= end_time:
        break

print(f"Fetched {len(all_klines)} candles total.")

# --- Convert to DataFrame ---
df = pd.DataFrame(all_klines, columns=[
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
])

df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

# Keep useful numeric columns (indices 1–10)
cols = ["open", "high", "low", "close", "volume",
        "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote"]
for c in cols:
    df[c] = df[c].astype(float).round(2)

# Optional: drop duplicates just in case
df = df.drop_duplicates(subset="open_time").reset_index(drop=True)

# --- Remove timestamp columns you don't need ---
df = df.drop(columns=["open_time", "close_time", "ignore"], errors="ignore")

# --- Save to JSON ---
df.to_json("btc_5m_week_3.json", orient="records", date_format="iso")
print("Success: Saved one week of 5m OHLC data to btc_5m_week.json")
