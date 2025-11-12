import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import random
import joblib




# 1, 42, 541, 703, 1729
# 7311994, 0xC0FFEE
# Setting a seed to eliminate randomness between runs:
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
production = True
BEST_EPOCH = 59
GRAPHS = True


# 1.) Load and pre-process the data
# Load JSON
#with open("btc_5m_week_3.json") as f:
with open("dnut_5m_recent2.json") as f:
#with open("tsla_5m_recent2.json") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)

# Optional: ensure correct numeric types
df = df.astype(float)



# ┌───┐
# |   |    
# └───┘
#  ╔══════════════════════════════════╗
#  ║Add Technical Indicators 10/8/'25 ║
#  ╚══════════════════════════════════╝

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
###########



# 10/15/25 -> Spot-derived Quantitative Sentiment
df['buy_ratio'] = df['taker_buy_base'] / df['volume']
df['buy_sell_imbalance'] = (2 * df['taker_buy_base'] - df['volume']) / df['volume']
df['volatility'] = (df['high'] - df['low']) / df['open']

#eps = 1e-8
#df['buy_ratio'] = df['taker_buy_base'] / df['volume'].replace(0, eps)
#df['buy_sell_imbalance'] = (2 * df['taker_buy_base'] - df['volume']) / df['volume'].replace(0, eps)

"""
### Creating target labels by computing exponential moving averages 
window = 10 
df['ema_close'] = df['close'].ewm(span=window, adjust=False).mean()

# Asking: "One candle from now, will the price be above the current EMA."
df['target'] = ((df['close'].shift(-1) - df['ema_close']) > 0).astype(int)


####
print(df['target'].value_counts(normalize=True))


# Normalize features
scaler = MinMaxScaler()

#10/8/'25
#feature_cols = ["open", "high", "low", "close", "volume",
#               "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote"]

feature_cols = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
    "macd", "macd_hist", "rsi", "atr",
    "bb_middle", "bb_upper", "bb_lower", "bb_width",
    "obv", "vwap"
]

#Drop NaN before scaling.
df.dropna(inplace=True)

# This is a data leakage pointV
scaled = scaler.fit_transform(df[feature_cols])
"""
if production == False:
    #Part 1:
    window = 10  # EMA span

    # ---------------------------
    # 1️⃣  Split FIRST (72/8/20)
    # ---------------------------
    n = len(df)
    train_end = int(n * 0.72)
    val_end   = int(n * 0.80)  # 72% + 8% = 80%

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    # Remove the NaNs before computing the Target.

    # Became necessary with the addition of Techincal Indicators
    # Must be done before creating sequences otherwise there could be a potential mismatch between X and Y.
    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
    # Became necessary with the addition of Spot-derived Quantitative Sentiment.
    val_df = val_df.replace([np.inf, -np.inf], np.nan).dropna()
    test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna()

    # ---------------------------
    # 2️⃣  Compute EMA SEPARATELY
    # ---------------------------
    for subset in [train_df, val_df, test_df]:
        subset['ema_close'] = subset['close'].ewm(span=window, adjust=False).mean()

    # ---------------------------
    # 3️⃣  Create Target (future direction)
    # ---------------------------
    for subset in [train_df, val_df, test_df]:
        subset['target'] = ((subset['close'].shift(-1) - subset['ema_close']) > 0).astype(int)
    subset.dropna(subset=['target'], inplace=True)

    # ---------------------------
    # 4️⃣  Scale USING TRAIN FIT ONLY
    # ---------------------------



    #10/16/'25
    #scaler = MinMaxScaler()
    feature_cols = ['close', 'ema_close']  # add others like MACD, RSI, etc.

    #train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    #val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
    #test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    # ---------------------------
    # ✅ 5️⃣ Ready for sequence creation
    # ---------------------------
    

    # Part2: 
    # Define which columns to scale
    # Chat put "ema_close", in here but I removed it.
    feature_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
        "macd", "macd_hist", "rsi", "atr",
        "bb_middle", "bb_upper", "bb_lower", "bb_width",
        "obv", "vwap",
        "buy_ratio", "buy_sell_imbalance", "volatility"
    ]

    # Initialize scaler
    scaler = MinMaxScaler()

    # Fit ONLY on training data
    scaler.fit(train_df[feature_cols])

    # Transform all sets using the same scaler
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    print(len(train_df), len(val_df), len(test_df))




    # 2.) Normalize Features
    def create_sequences(features, labels, seq_len=60):
        X, y = [], []
        for i in range(len(features) - seq_len):
            X.append(features[i:i+seq_len])
            y.append(labels[i+seq_len])
        return np.array(X), np.array(y)
    
    SEQ_LEN = 48

    print("NaNs in train_df:", train_df.isna().sum().sum())
    print("Infs in train_df:", np.isinf(train_df.values).sum())

    print("NaNs in val_df:", val_df.isna().sum().sum())
    print("Infs in val_df:", np.isinf(val_df.values).sum())

    print("NaNs in test_df:", test_df.isna().sum().sum())
    print("Infs in test_df:", np.isinf(test_df.values).sum())

    print("Target NaNs:", np.isnan(train_df["target"]).sum())
    print("Target Infs:", np.isinf(train_df["target"]).sum())

    # Create sequences separately for each split (leak-free)
    X_train, y_train = create_sequences(train_df[feature_cols].values, train_df["target"].values, SEQ_LEN)
    X_val, y_val     = create_sequences(val_df[feature_cols].values, val_df["target"].values, SEQ_LEN)
    X_test, y_test   = create_sequences(test_df[feature_cols].values, test_df["target"].values, SEQ_LEN)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

else:
    # ===============================
    # 3. Split FIRST (80/20)
    # ===============================
    n = len(df)
    train_end = int(n * 0.8)
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[train_end:].copy()

    # 10/15/'25
    # Remove the NaNs before computing the target.
    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
    test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna()

    # ===============================
    # 4. Compute EMA + Target AFTER Split (no leakage)
    # ===============================
    window = 10
    for subset in [train_df, test_df]:
        subset['ema_close'] = subset['close'].ewm(span=window, adjust=False).mean()
        subset['target'] = ((subset['close'].shift(-1) - subset['ema_close']) > 0).astype(int)
        subset.dropna(subset=['target'], inplace=True)

    # ===============================
    # 5. Scale using TRAIN fit only
    # ===============================
    feature_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
        "macd", "macd_hist", "rsi", "atr",
        "bb_middle", "bb_upper", "bb_lower", "bb_width",
        "obv", "vwap",
        "buy_ratio", "buy_sell_imbalance", "volatility"
    ]

    # Initially I was removing the NaNs here - after computing the target.
    

    scaler = MinMaxScaler()
    #train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
    scaler.fit(train_df[feature_cols])

    # 10/11/25
    # === SAVE THE SCALER ===
    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved as scaler.pkl")

    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # ===============================
    # 6. Sequence creation
    # ===============================
    def create_sequences(features, labels, seq_len=60):
        X, y = [], []
        for i in range(len(features) - seq_len):
            X.append(features[i:i+seq_len])
            y.append(labels[i+seq_len])
        return np.array(X), np.array(y)

    SEQ_LEN = 48
    X_train, y_train = create_sequences(train_df[feature_cols].values, train_df["target"].values, SEQ_LEN)
    X_test, y_test = create_sequences(test_df[feature_cols].values, test_df["target"].values, SEQ_LEN)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")


"""
#SEQ_LEN = 60      10/8/'25
SEQ_LEN = 48
X, y = create_sequences(scaled, df["target"].values, SEQ_LEN)


#split = int(len(X) * 0.8)
#X_train, X_test = X[:split], X[split:]
#y_train, y_test = y[:split], y[split:]

# This all translates to:    72%  -> Train, 
#                             8%  ->  Validation, 
#                            20% -> Test
split = int(len(X) * 0.8)    # 0.8X       = 0.8X
val_split = int(split * 0.9) # 0.8X * 0.9 = 0.72X

X_train, X_val, X_test = X[:val_split], X[val_split:split], X[split:]
y_train, y_val, y_test = y[:val_split], y[val_split:split], y[split:]
"""




# Figure out if there is class imbalance or not.
#print(y_train.mean())



#  ╔══════════════════════════════════╗
#  ║3.) Train the Model   10/12/'25   ║
#  ╚══════════════════════════════════╝

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.regularizers import l2

# Could try adding dropout after relu                                  [X]
# *Note: For seed 42 - it went to epoch 17 without an improvement.

# Could try 64, 32, 16 instead of doing 32, 16, 8.                     [X]
# *Note: Only tested on one, but loss went from 0.6633 to 0.6764.

# Could try stacking 1D-CNN with the LSTM                              [X]
# *Note: Only tested on one, but loss went from 0.6633 to 0.6702.

# Could try adding "Market Sentiment"                                  []


model = Sequential([

    #LSTM(32, return_sequences=True, input_shape=(SEQ_LEN, len(feature_cols))),
    LSTM(32, return_sequences=True, recurrent_dropout=0.2, input_shape=(SEQ_LEN, len(feature_cols))),
    Dropout(0.2),
    

    #LSTM(16),  # smaller layer summarizing the sequence
    #LSTM(16, recurrent_dropout=0.2),
    LSTM(16, recurrent_dropout=0.2),
    Dropout(0.2),

    #Dense(8, activation='relu'),   # a small dense layer
    #Dropout(0.2),

    Dense(1, activation='sigmoid')  # binary up/down output
])


"""
10/29/25
inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))

# Layer normalization
x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)

# Multi-head self-attention
attn_output = tf.keras.layers.MultiHeadAttention(
    num_heads=4, key_dim=64, dropout=0.2
)(x, x)
x = tf.keras.layers.Add()([x, attn_output])

# Layer norm before FFN
x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

# Feed-forward network
ffn = tf.keras.layers.Dense(128, activation="relu")(x)
ffn = tf.keras.layers.Dropout(0.2)(ffn)
ffn = tf.keras.layers.Dense(64, activation="relu")(ffn)

# Project x to same dimension as ffn so Add() works
x_proj = tf.keras.layers.Dense(64)(x)
x = tf.keras.layers.Add()([x_proj, ffn])

# Output head
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
"""

"""
10/14/25
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
model = Sequential([
    # --- 1D CNN feature extractor ---
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
           input_shape=(SEQ_LEN, len(feature_cols))),
    MaxPooling1D(pool_size=2, padding='same'),  # reduces sequence length by ~2
    Dropout(0.2),

    # --- LSTM layers ---
    LSTM(32, return_sequences=True, recurrent_dropout=0.2),
    Dropout(0.2),
    LSTM(16, recurrent_dropout=0.2),
    Dropout(0.2),

    # --- Dense head ---
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # binary up/down output
])
"""


# Was 0.0009
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4.) Train the model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


if production == False:
    # This part allows for early stopping while training with more epochs (100 up from 20).
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True)
    
    check = ModelCheckpoint('best_lstm_model.h5', save_best_only=True)

    # 10/13/'25 _Testing
    reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    # what to watch — usually val_loss or val_accuracy
    factor=0.5,            # new_lr = old_lr * factor
    patience=5,            # epochs to wait before reducing
    #patience=10,
    min_lr=1e-6,           # don’t go below this
    verbose=1              # print when it happens
    )
else:
    reduce_lr = ReduceLROnPlateau(
    monitor='loss',        # what to watch — usually val_loss or val_accuracy
    factor=0.5,            # new_lr = old_lr * factor
    patience=5,            # epochs to wait before reducing
    #patience=10,
    min_lr=1e-6,           # don’t go below this
    verbose=1              # print when it happens
    )
    

### Attempting to fix everything trending up. ###############
# Note: Adding more data fixed this problem. I am keeping the weights part because it helps
# with performance.
from sklearn.utils import class_weight

# Check your classes explicitly
classes = np.unique(y_train)
print(np.unique(y_test))
print(np.unique(y_train))
print("Classes found:", classes)


# Compute base weights
base_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

base_weights = dict(zip(classes, base_weights))

#Experiment 10/30 -> Trying to figure out what is going on with the Raspberry Pi.
#base_weights = {0: 1.1558704453441295, 1: 0.8811728395061729}

print("Adjusted weights:", base_weights)



# For reduce_lr -> I would like to incorporate this. I think it can help. For instance on seed 1729 -
# The best epoch is 56. However how do I incoporate it within the 80/20 split? I used 'loss' as a monitor
# and it brought the test loss down to 0.6642 when it was originally at 0.6830 for just the 0.001 LR.
# 
# According to ChatGPT this should be fine. 10/13/'25 

if production == False:
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        #validation_split=0.1,
        validation_data = (X_val, y_val),
        shuffle=False,
        #callbacks=[reduce_lr, early_stop, check],
        callbacks=[check],
        class_weight=base_weights
    )
else:
    history = model.fit(
    X_train, y_train,
    epochs=BEST_EPOCH,
    batch_size=64,
    shuffle=False,
    class_weight=base_weights,
    #callbacks= reduce_lr
)



from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, log_loss, classification_report
import matplotlib.pyplot as plt

if production == False and GRAPHS == True:
    # 5.) Predict and interpret probabilities

    
    # Load the best model
    #model = load_model('best_lstm_model.h5')


    probs = model.predict(X_test).flatten()

    upper = np.percentile(probs, 60)
    lower = np.percentile(probs, 40)

    df_result = pd.DataFrame({
        "prob_up": probs,
        "actual": y_test
    })

    df_result["signal"] = np.where(df_result["prob_up"] > upper, "BUY",
                        np.where(df_result["prob_up"] < lower, "SELL", "HOLD"))

    print(f"Dynamic thresholds -> upper: {upper:.3f}, lower: {lower:.3f}")


    #from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, log_loss

    #####
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Final Test Loss: {loss:.4f}")

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))



    # Trying this. It compares the y_test value (actual) to the dynamic value 50th percentile.
    goal = np.percentile(probs, 50)

    # threshold at 0.5 or your dynamic levels
    # Dynamic 50th percentile threshold metrics
    y_pred_dynamic = (probs > goal).astype(int)
    print("\n=== Dynamic 50th Percentile Threshold ===")
    print("Dynamic Accuracy:", accuracy_score(y_test, y_pred_dynamic))
    print("Dynamic Log Loss:", log_loss(y_test, probs))  # still use raw probs for log_loss
    print("Dynamic Precision:", precision_score(y_test, y_pred_dynamic))
    print("Dynamic Recall:", recall_score(y_test, y_pred_dynamic))
    print("Dynamic F1:", f1_score(y_test, y_pred_dynamic))



    y_pred_prob = model.predict(X_test).flatten()


    

    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.show()

    # Accuracy over epochs
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Loss over epochs.
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    #10/25/25: Previously it was just one graph of the entire dataframe before any invalid frames were dropped.
    # Additionally, I just noticed todat that the actual USD values are not displayed.
    # Closing Price Only.
    plt.figure(figsize=(12, 6))
    #plt.plot(df['close'].values, label='Closing Price', color='blue') <------- 10/25/25
    plt.plot(train_df['close'].values, label='Closing Price', color='blue')
    plt.title('Closing Price Over Time')
    plt.xlabel('Time (5-minute intervals)')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Closing Price Only.
    plt.figure(figsize=(12, 6))
    plt.plot(val_df['close'].values, label='Closing Price', color='blue')
    plt.title('Closing Price Over Time')
    plt.xlabel('Time (5-minute intervals)')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Closing Price Only.
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['close'].values, label='Closing Price', color='blue')
    plt.title('Closing Price Over Time')
    plt.xlabel('Time (5-minute intervals)')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()



    # Dynamic Scatter Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(df_result)), df_result["prob_up"], 
                c=df_result["actual"], cmap='coolwarm', alpha=0.6)
    plt.axhline(upper, color='green', linestyle='--', label=f'Upper threshold ({upper:.2f})')
    plt.axhline(lower, color='red', linestyle='--', label=f'Lower threshold ({lower:.2f})')
    plt.title("Predicted Probability of Up Move (Dynamic Thresholds)")
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Probability (prob_up)")
    plt.legend()
    plt.show()

    # Creates a scatter plot that shows predictions, actual market behavior, and thresholds.
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test)), y_pred_prob, c=y_test, cmap='bwr', alpha=0.6)
    plt.axhline(0.5, color='k', linestyle='--', label='0.5 Threshold')
    plt.axhline(0.6, color='g', linestyle='--', label='Buy (0.6)')
    plt.axhline(0.4, color='r', linestyle='--', label='Sell (0.4)')
    plt.title('Predicted Probability vs Actual Trend')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Probability (Price Up)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Also print a few summary stats
    print("Mean probability for actual UPs:", y_pred_prob[y_test == 1].mean())
    print("Mean probability for actual DOWNs:", y_pred_prob[y_test == 0].mean())


elif production == True and GRAPHS == True:
    probs = model.predict(X_test).flatten()
    upper = np.percentile(probs, 60)
    lower = np.percentile(probs, 40)

    # Fix this later.##################
    y_pred_prob = probs  

    df_result = pd.DataFrame({
        "prob_up": probs,
        "actual": y_test
    })
    df_result["signal"] = np.where(df_result["prob_up"] > upper, "BUY",
                            np.where(df_result["prob_up"] < lower, "SELL", "HOLD"))

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Final Test Loss: {loss:.4f}")

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    goal = np.percentile(probs, 50)
    y_pred_dynamic = (probs > goal).astype(int)
    print("\n=== Dynamic 50th Percentile Threshold ===")
    print("Dynamic Accuracy:", accuracy_score(y_test, y_pred_dynamic))
    print("Dynamic Log Loss:", log_loss(y_test, probs))
    print("Dynamic Precision:", precision_score(y_test, y_pred_dynamic))
    print("Dynamic Recall:", recall_score(y_test, y_pred_dynamic))
    print("Dynamic F1:", f1_score(y_test, y_pred_dynamic))

    # ===============================
    # 10. Plots
    # ===============================
    

    plt.figure(figsize=(10,4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Assuming:
    #   df_result      = your full dataset or results DataFrame
    #   y_test         = true labels from the 20% test split
    #   y_pred_prob    = predicted probabilities (sigmoid outputs) for the test set
    #   upper, lower   = your dynamic threshold values (floats)

    # Create an index that corresponds only to the test portion
    test_index = range(len(y_test))

    # --- Dynamic Scatter Plot (Test Data Only) ---
    plt.figure(figsize=(10, 5))
    plt.scatter(test_index, y_pred_prob, 
              c=y_test, cmap='coolwarm', alpha=0.6)
    plt.axhline(upper, color='green', linestyle='--', label=f'Upper threshold ({upper:.2f})')
    plt.axhline(lower, color='red', linestyle='--', label=f'Lower threshold ({lower:.2f})')
    plt.title("Predicted Probability of Up Move (Test Set Only)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Predicted Probability (prob_up)")
    plt.legend()
    plt.show()

    # --- Predicted Probability vs Actual Trend (Test Data Only) ---
    plt.figure(figsize=(8, 6))
    plt.scatter(test_index, y_pred_prob, c=y_test, cmap='bwr', alpha=0.6)
    plt.axhline(0.5, color='k', linestyle='--', label='0.5 Threshold')
    plt.axhline(0.6, color='g', linestyle='--', label='Buy (0.6)')
    plt.axhline(0.4, color='r', linestyle='--', label='Sell (0.4)')
    plt.title('Predicted Probability vs Actual Trend (Test Set Only)')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Predicted Probability (Price Up)')
    plt.legend()
    plt.grid(True)
    plt.show()

