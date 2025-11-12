# 10/29/'25 This was all done quickly. Make sure that everything matches in the model.


import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# === SETTINGS ===
SEQ_LENGTH = 48
EMA_WINDOW = 10
SYMBOL = "TSLA"
INTERVAL = "5m"
NUM_CANDLES = 148

# =====================================================
# === FETCH LATEST TSLA 5-MINUTE DATA (via yfinance)
# =====================================================

end = datetime.utcnow()
start = end - timedelta(days=60)  # yfinance 5m limit = 60 days

print(f"Fetching {SYMBOL} {INTERVAL} data from {start} to {end}")
df = yf.download(SYMBOL, interval=INTERVAL, start=start, end=end, progress=False)

if df.empty:
    raise SystemExit("❌ No data returned. Try shorter range or different interval.")

# Flatten columns if needed
df = df.reset_index()
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join([c for c in col if c]).strip().lower() for col in df.columns.values]
else:
    df.columns = [c.lower() for c in df.columns]

# Detect datetime column
datetime_col = next((c for c in df.columns if "time" in c or "date" in c), None)
if not datetime_col:
    raise ValueError("No datetime column found!")

# Standardize column names
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
df.rename(columns=rename_map, inplace=True)

# Ensure columns exist
required = ["open_time", "open", "high", "low", "close", "volume"]
for col in required:
    if col not in df.columns:
        raise KeyError(f"Missing expected column: {col}")

# Standardize format
df["open_time"] = pd.to_datetime(df["open_time"]).dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
df = df[required]

# Match your 10/22/'25 file: Add zero placeholders for Binance-style fields
df["quote_asset_volume"] = 0.0
df["num_trades"] = 0.0
df["taker_buy_base"] = 0.0
df["taker_buy_quote"] = 0.0

# Drop open_time just like your original preprocessing
df.drop("open_time", axis=1, inplace=True)

# Keep last NUM_CANDLES
df = df.tail(NUM_CANDLES).copy()

# Save for debugging / transparency
df.to_json("live5m.json", orient="records", date_format="iso")
print(f"✅ Saved {len(df)} rows to live5m.json (TSLA 5m data)")
print(f"   Range: {start} → {end}")

# =====================================================
# === LOAD MODEL & SCALER ===
# =====================================================
model = load_model("best_lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# =====================================================
# === LOAD NEW DATA (same schema as training) ===
# =====================================================
df = pd.read_json("live5m.json")
df = df.astype(float)

# =====================================================
# === FEATURE ENGINEERING (same as training) ===
# =====================================================

# MACD
short_window = 12
long_window = 26
signal_window = 9
df['ema_short'] = df['close'].ewm(span=short_window, adjust=False).mean()
df['ema_long'] = df['close'].ewm(span=long_window, adjust=False).mean()
df['macd'] = df['ema_short'] - df['ema_long']
df['macd_signal'] = df['macd'].ewm(span=signal_window, adjust=False).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']

# RSI
rsi_window = 14
delta = df['close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=rsi_window, min_periods=1).mean()
avg_loss = pd.Series(loss).rolling(window=rsi_window, min_periods=1).mean()
rs = avg_gain / (avg_loss + 1e-10)
df['rsi'] = 100 - (100 / (1 + rs))

# ATR
atr_window = 14
high_low = df['high'] - df['low']
high_close = np.abs(df['high'] - df['close'].shift())
low_close = np.abs(df['low'] - df['close'].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr'] = true_range.rolling(window=atr_window, min_periods=1).mean()

# Bollinger Bands
bb_window = 20
df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
df['bb_std'] = df['close'].rolling(window=bb_window).std()
df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
df['bb_width'] = df['bb_upper'] - df['bb_lower']

# OBV
df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

# VWAP
df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

# Drop unnecessary
df.drop(['ema_short', 'ema_long', 'macd_signal', 'bb_std'], axis=1, inplace=True)

# EMA for smoothing
df['ema_close'] = df['close'].ewm(span=EMA_WINDOW, adjust=False).mean()
df.dropna(inplace=True)

df['buy_ratio'] = df['taker_buy_base'] / df['volume']
df['buy_sell_imbalance'] = (2 * df['taker_buy_base'] - df['volume']) / df['volume']
df['volatility'] = (df['high'] - df['low']) / df['open']


# =====================================================
# === SCALE AND PREDICT ===
# =====================================================
feature_cols = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
    "macd", "macd_hist", "rsi", "atr",
    "bb_middle", "bb_upper", "bb_lower", "bb_width",
    "obv", "vwap", "buy_ratio", "buy_sell_imbalance", "volatility"
]
df_scaled = pd.DataFrame(scaler.transform(df[feature_cols]), columns=feature_cols)

last_seq = df_scaled.values[-SEQ_LENGTH:]
X_live = np.expand_dims(last_seq, axis=0)

y_pred = model.predict(X_live)
prob = float(y_pred[0][0])

if prob > 0.5:
    print(f"📈 Predicted UP (prob = {prob:.4f})")
else:
    print(f"📉 Predicted DOWN (prob = {prob:.4f})")
