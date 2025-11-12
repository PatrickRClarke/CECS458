# 10/22/'25

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


symbol = "TSLA"
interval = "5m"

end = datetime.utcnow()
start = end - timedelta(days=60)  # within Yahoo's 60-day limit

print(f"Fetching {symbol} {interval} data from {start} to {end}")
df = yf.download(symbol, interval=interval, start=start, end=end, progress=True)

if df.empty:
    raise SystemExit("❌ No data returned. Try shorter range or different interval.")

# Reset index and flatten columns if necessary
df = df.reset_index()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join([c for c in col if c]).strip().lower() for col in df.columns.values]
else:
    df.columns = [c.lower() for c in df.columns]

# Detect possible datetime column name
datetime_col = next((c for c in df.columns if "time" in c or "date" in c), None)
if not datetime_col:
    raise ValueError("No datetime column found!")

# Standardize column names (case-insensitive)
rename_map = {
    datetime_col: "open_time",
    "open_tsla": "open",
    "high_tsla": "high",
    "low_tsla": "low",
    "close_tsla": "close",
    "volume_tsla": "volume",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}

df = df.rename(columns=rename_map)

# Ensure required columns exist
required = ["open_time", "open", "high", "low", "close", "volume"]
for col in required:
    if col not in df.columns:
        raise KeyError(f"Missing expected column: {col}")

# Convert to Binance-style
df["open_time"] = pd.to_datetime(df["open_time"]).dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
df = df[required]

# Add Binance-style placeholders
df["quote_asset_volume"] = 0.0
df["num_trades"] = 0.0
df["taker_buy_base"] = 0.0
df["taker_buy_quote"] = 0.0


# Fixing ChatGPTs failure.
df.drop('open_time', axis=1, inplace=True)

# Save as JSON
output_file = "tsla_5m_recent3.json"
df.to_json(output_file, orient="records", date_format="iso")

print(f"✅ Success: Saved {len(df)} rows of 5m OHLC data to {output_file}")
#print(df.head(2))
