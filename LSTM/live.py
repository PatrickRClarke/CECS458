import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

SEQ_LENGTH = 48
EMA_WINDOW = 10

# 10/29/'25: One thing I still need to do is ensure that when a model is trained that it is trained clear
# up to where the data here would begin initially.  (Present-148 5m candles.)
# //////////////////////////////////////////////////////
import requests
import pandas as pd
from datetime import datetime

# --- Settings ---
symbol = "BTCUSDT"
interval = "5m"
num_candles = 148  # fixed number of candles to fetch

# Binance US base URL
BASE_URL = "https://api.binance.us/api/v3/klines"

# --- Fetch last 148 candles ---
response = requests.get(BASE_URL, params={"symbol": symbol, "interval": interval, "limit": 148})
data = response.json()
if not data:
    raise ValueError("No data returned from Binance API")

df = pd.DataFrame(data, columns=[
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
])

# Convert timestamps
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

# Save timestamps for printing/logging
start_ts = df["open_time"].iloc[0]
end_ts = df["open_time"].iloc[-1]

# Keep only numeric columns for JSON / LSTM
cols = ["open", "high", "low", "close", "volume",
        "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote"]
df_numeric = df[cols].astype(float).round(2)
df_numeric.to_json("live5m.json", orient="records", date_format="iso")

print(f"Success: Fetched {len(df_numeric)} recent 5-minute candles from Binance")
print(f"From {start_ts} to {end_ts}")



#/////////////////////////////////////

# === LOAD MODEL & SCALER ===
model = load_model("best_lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# === LOAD NEW DATA (same schema as training) ===
df = pd.read_json("live5m.json")
df = df.astype(float)

# === RECREATE TECHNICAL INDICATORS (same code as training) ===
# --- MACD (Moving Average Convergence Divergence) ---
short_window = 12
long_window = 26
signal_window = 9
df['ema_short'] = df['close'].ewm(span=short_window, adjust=False).mean()
df['ema_long'] = df['close'].ewm(span=long_window, adjust=False).mean()
df['macd'] = df['ema_short'] - df['ema_long']
df['macd_signal'] = df['macd'].ewm(span=signal_window, adjust=False).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']

# --- RSI (Relative Strength Index) ---
rsi_window = 14
delta = df['close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=rsi_window, min_periods=1).mean()
avg_loss = pd.Series(loss).rolling(window=rsi_window, min_periods=1).mean()
rs = avg_gain / (avg_loss + 1e-10)
df['rsi'] = 100 - (100 / (1 + rs))

# --- ATR (Average True Range) ---
atr_window = 14
high_low = df['high'] - df['low']
high_close = np.abs(df['high'] - df['close'].shift())
low_close = np.abs(df['low'] - df['close'].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr'] = true_range.rolling(window=atr_window, min_periods=1).mean()

# --- Bollinger Bands ---
bb_window = 20
df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
df['bb_std'] = df['close'].rolling(window=bb_window).std()
df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
df['bb_width'] = df['bb_upper'] - df['bb_lower']

# --- OBV (On-Balance Volume) ---
df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

# --- VWAP (Volume Weighted Average Price) ---
df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

# Drop intermediate columns you don’t want to feed directly into the model
df.drop(['ema_short', 'ema_long', 'macd_signal', 'bb_std'], axis=1, inplace=True)




# Compute EMA for target alignment
df['ema_close'] = df['close'].ewm(span=EMA_WINDOW, adjust=False).mean()
df.dropna(inplace=True)

# === SCALE ===
feature_cols = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
    "macd", "macd_hist", "rsi", "atr",
    "bb_middle", "bb_upper", "bb_lower", "bb_width",
    "obv", "vwap"
]
df_scaled = pd.DataFrame(scaler.transform(df[feature_cols]), columns=feature_cols)

# === PREPARE LAST SEQUENCE ===
last_seq = df_scaled.values[-SEQ_LENGTH:]
X_live = np.expand_dims(last_seq, axis=0)

# === PREDICT ===
y_pred = model.predict(X_live)
prob = float(y_pred[0][0])

if prob > 0.5:
    print(f"📈 Predicted UP (prob = {prob:.4f})")
else:
    print(f"📉 Predicted DOWN (prob = {prob:.4f})")