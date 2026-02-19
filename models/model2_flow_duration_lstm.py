"""
model2_flow_duration_lstm.py
=============================
Model 2 — LSTM Time Series Predictor
Target variable : flow_duration  (continuous regression)
Dataset         : RT-IoT2022  (UCI ML Repository, ID 942)

NOTE: This model predicts FLOW DURATION — a continuous temporal variable —
      using sequences of past network flow features.  This is intentionally
      DIFFERENT from Model 1, which classifies the categorical variable
      Attack_type.

Approach
--------
We treat consecutive network flow records as a time series.  A sliding
window of SEQ_LEN past flow feature vectors is used to predict the
flow_duration of the next (t+1) flow.  This mimics the operational scenario
where a network monitor must anticipate how long the next connection will last,
which can be used to flag anomalously long or short flows.

Architecture (built from scratch with TensorFlow/Keras):
    Input (SEQ_LEN × n_features)
        → LSTM(64, return_sequences=True)
        → Dropout(0.3)
        → LSTM(32, return_sequences=False)
        → Dropout(0.2)
        → Dense(16, activation='relu')
        → Dense(1)              ← scalar flow_duration prediction

Usage:
    pip install ucimlrepo tensorflow scikit-learn pandas numpy
    python model2_flow_duration_lstm.py
"""

# =============================================================================
# Imports
# =============================================================================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Fix random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    # Sliding-window parameters
    "seq_len":        20,     # number of past flows used as input sequence
    # Train / test split
    "test_size":      0.20,
    "val_size":       0.10,
    # Training
    "batch_size":     256,
    "epochs":         50,
    "patience":       7,
    "lr":             1e-3,
    "lr_decay":       0.5,
    "lr_patience":    3,
    # Architecture
    "lstm_units":     [64, 32],   # units in LSTM layer 1 and 2
    "dense_units":    16,          # hidden dense before output
    "dropout_rates":  [0.3, 0.2], # per LSTM layer
    # Output
    "output_csv":      "../outputs/model2_predictions.csv",
    "model_save_path": "../models/model2_flow_duration_lstm_saved.keras",
}

# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data() -> pd.DataFrame:
    """
    Fetch the RT-IoT2022 dataset from the UCI ML Repository.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of features + target column 'Attack_type'
        (kept for reference / potential filtering).
    """
    print("[1/6] Loading RT-IoT2022 dataset from UCI repository …")
    from ucimlrepo import fetch_ucirepo

    rt_iot2022 = fetch_ucirepo(id=942)
    X_raw = rt_iot2022.data.features
    y_raw = rt_iot2022.data.targets

    df = X_raw.copy()
    df["Attack_type"] = y_raw.values.ravel()

    print(f"    Dataset shape : {df.shape}")
    return df


# =============================================================================
# 2. PREPROCESSING FOR TIME SERIES
# =============================================================================

def preprocess(df: pd.DataFrame):
    """
    Prepare the dataset for LSTM-based time series regression.

    Design decisions
    ----------------
    * The target is `flow_duration` (column already in the features).
    * We use all numeric network-flow features as the input sequence —
      this gives the LSTM temporal context about packet rates, byte sizes,
      inter-arrival times, and TCP flags leading up to each flow.
    * Categorical columns (proto, service, Attack_type) are dropped to keep
      inputs purely numeric and avoid leakage of label information.
    * Rows are kept in their original order, which preserves the implicit
      temporal ordering of captured network flows.
    * Sliding window construction: for each position t in [SEQ_LEN, N],
      X[t] = feature matrix[t-SEQ_LEN : t]   (shape: SEQ_LEN × n_features)
      y[t] = flow_duration[t]                  (scalar target)

    Returns
    -------
    tuple : X_train, X_val, X_test, y_train, y_val, y_test,
            target_scaler (for inverse-transforming predictions)
    """
    print("[2/6] Preprocessing for time series …")

    # --- Drop categorical columns (retain numeric flow statistics only) ---
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df_num = df.drop(columns=cat_cols)

    # Impute any NaN with column median
    df_num = df_num.fillna(df_num.median(numeric_only=True))

    # --- Identify target column ---
    target_col = "flow_duration"
    if target_col not in df_num.columns:
        # Fallback: use the first numeric column as target
        target_col = df_num.columns[0]
        print(f"    Warning: 'flow_duration' not found; using '{target_col}'")

    feature_cols = [c for c in df_num.columns if c != target_col]

    print(f"    Target column  : {target_col}")
    print(f"    Feature count  : {len(feature_cols)}")

    # --- Scale features and target separately ---
    # Feature scaler: applied to the n_features input dimensions
    feat_scaler = StandardScaler()
    features_scaled = feat_scaler.fit_transform(df_num[feature_cols].values)

    # flow_duration spans ~10 orders of magnitude (min≈0, max≈21728 s).
    # A log1p transform compresses this extreme range before standardisation,
    # allowing the LSTM to learn meaningful patterns without being dominated
    # by rare very-long flows. Metrics are reported in log1p-space (which is
    # interpretable: RMSE of 0.1 log-seconds ≈ ~10 % relative error).
    target_log = np.log1p(df_num[target_col].values)

    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(
        target_log.reshape(-1, 1)
    ).ravel()

    # --- Build sliding-window sequences ---
    SEQ = CONFIG["seq_len"]
    n = len(features_scaled)

    print(f"    Building sliding windows (seq_len={SEQ}) …")
    X_seq = np.stack(
        [features_scaled[i : i + SEQ] for i in range(n - SEQ)],
        axis=0
    )                            # shape: (n-SEQ, SEQ, n_features)
    y_seq = target_scaled[SEQ:]  # shape: (n-SEQ,)

    print(f"    Sequence dataset shape: X={X_seq.shape}, y={y_seq.shape}")

    # --- Train / test split (no shuffle to preserve temporal order) ---
    split_test  = int(len(X_seq) * (1 - CONFIG["test_size"]))
    X_tr, X_test = X_seq[:split_test],  X_seq[split_test:]
    y_tr, y_test = y_seq[:split_test],  y_seq[split_test:]

    # --- Train / validation split (again no shuffle) ---
    split_val = int(len(X_tr) * (1 - CONFIG["val_size"]))
    X_train, X_val = X_tr[:split_val], X_tr[split_val:]
    y_train, y_val = y_tr[:split_val], y_tr[split_val:]

    print(f"    Train / Val / Test : {len(X_train)} / "
          f"{len(X_val)} / {len(X_test)} sequences\n")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            target_scaler, target_col, feat_scaler)


# =============================================================================
# 3. MODEL ARCHITECTURE
# =============================================================================

def build_model(seq_len: int, n_features: int) -> keras.Model:
    """
    Build a stacked LSTM regression model from scratch.

    Architecture rationale
    ----------------------
    * LSTM Layer 1 (64 units, return_sequences=True):
        Processes the full SEQ_LEN time steps and passes its hidden state
        at every step to Layer 2 — enabling the second LSTM to see the
        full temporal context built up by the first.
    * Dropout after LSTM 1 (rate=0.3):
        Regularises the recurrent connections to prevent memorisation of
        specific flow sequences.
    * LSTM Layer 2 (32 units, return_sequences=False):
        Summarises the full sequence into a single context vector,
        capturing long-range temporal dependencies across 20 flows.
    * Dropout after LSTM 2 (rate=0.2):
        Additional regularisation on the compressed representation.
    * Dense(16, ReLU):
        Non-linear projection from the LSTM summary to a compact space
        before the scalar output.
    * Dense(1):
        Scalar output — the predicted (standardised) flow_duration.
        No activation function → unconstrained regression output.

    Parameters
    ----------
    seq_len    : int — length of the input sequence (number of past flows)
    n_features : int — number of numeric features per time step

    Returns
    -------
    keras.Model (uncompiled)
    """
    inputs = keras.Input(shape=(seq_len, n_features), name="flow_sequence")

    # --- LSTM Layer 1: processes all time steps, passes full sequence ---
    x = layers.LSTM(
        CONFIG["lstm_units"][0],
        return_sequences=True,   # pass hidden state at every step to next layer
        name="lstm_1"
    )(inputs)
    x = layers.Dropout(CONFIG["dropout_rates"][0], name="drop_lstm_1")(x)

    # --- LSTM Layer 2: condenses sequence into a single context vector ---
    x = layers.LSTM(
        CONFIG["lstm_units"][1],
        return_sequences=False,  # only return the final hidden state
        name="lstm_2"
    )(x)
    x = layers.Dropout(CONFIG["dropout_rates"][1], name="drop_lstm_2")(x)

    # --- Dense projection layer ---
    x = layers.Dense(CONFIG["dense_units"], activation="relu",
                     name="dense_hidden")(x)

    # --- Scalar output: predicted flow_duration (in standardised units) ---
    outputs = layers.Dense(1, name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name="lstm_flow_duration_predictor")
    return model


# =============================================================================
# 4. TRAINING
# =============================================================================

def train_model(model: keras.Model,
                X_train, y_train,
                X_val, y_val) -> keras.callbacks.History:
    """
    Compile and train the LSTM with MSE loss and Adam optimiser.

    Loss: Mean Squared Error — standard for regression tasks; penalises
          large prediction errors more than MAE, which is appropriate here
          because unusually long flows (potential attacks) should not be
          under-penalised.

    Parameters
    ----------
    model             : uncompiled keras.Model
    X_train, y_train  : training sequences and scalar targets
    X_val, y_val      : validation sequences and scalar targets

    Returns
    -------
    keras.callbacks.History
    """
    print("[3/6] Compiling and training LSTM …")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["lr"]),
        loss="mse",              # minimise mean squared error
        metrics=["mae"]          # track mean absolute error as auxiliary metric
    )

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CONFIG["patience"],
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=CONFIG["lr_decay"],
            patience=CONFIG["lr_patience"],
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=CONFIG["model_save_path"],
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        callbacks=callbacks,
        verbose=1
    )

    return history


# =============================================================================
# 5. EVALUATION
# =============================================================================

def evaluate_model(model: keras.Model,
                   X_test, y_test,
                   target_scaler: StandardScaler,
                   target_col: str) -> dict:
    """
    Evaluate the LSTM predictor on the held-out test set.

    Metrics
    -------
    - RMSE in original units (inverse-transformed)
    - MAE  in original units
    - R²   coefficient of determination

    Parameters
    ----------
    model          : trained keras.Model
    X_test, y_test : test sequences and standardised targets
    target_scaler  : fitted StandardScaler for the target column
    target_col     : column name string (for display only)

    Returns
    -------
    dict with scalar metrics and prediction arrays
    """
    print("[4/6] Evaluating on test set …")

    # Predict in standardised space
    y_pred_scaled = model.predict(
        X_test, batch_size=CONFIG["batch_size"], verbose=0
    ).ravel()

    # Inverse-transform from standardised → log1p space
    y_pred_log = target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()
    y_true_log = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).ravel()

    # Metrics in log1p space (primary — robust to outliers)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log  = mean_absolute_error(y_true_log, y_pred_log)
    r2_log   = r2_score(y_true_log, y_pred_log)

    # Back-transform to original seconds for interpretability
    y_pred_orig = np.expm1(y_pred_log)
    y_true_orig = np.expm1(y_true_log)

    print(f"\n    Target          : {target_col}")
    print(f"    --- Log1p space (primary metrics) ---")
    print(f"    Test RMSE (log1p): {rmse_log:.4f}")
    print(f"    Test MAE  (log1p): {mae_log:.4f}")
    print(f"    Test R²   (log1p): {r2_log:.4f}")
    print(f"    --- Original scale (seconds) ---")
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae_orig  = mean_absolute_error(y_true_orig, y_pred_orig)
    print(f"    Test RMSE (s)    : {rmse_orig:.4f}")
    print(f"    Test MAE  (s)    : {mae_orig:.4f}\n")

    return {
        "rmse": rmse_log, "mae": mae_log, "r2": r2_log,
        "rmse_orig": rmse_orig, "mae_orig": mae_orig,
        "y_pred": y_pred_orig, "y_true": y_true_orig,
    }


# =============================================================================
# 6. SAVE PREDICTIONS
# =============================================================================

def save_predictions(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     target_col: str) -> None:
    """
    Write test-set predictions to CSV for dashboard consumption.

    Output columns
    --------------
    true_flow_duration  : actual observed value
    predicted_flow_duration : model's prediction
    absolute_error      : |true - predicted|  (useful for anomaly flagging)

    Parameters
    ----------
    y_true     : array of ground-truth flow_duration values
    y_pred     : array of predicted flow_duration values
    target_col : column name (used to label CSV columns)
    """
    print("[5/6] Saving predictions …")

    results_df = pd.DataFrame({
        f"true_{target_col}":      y_true.round(4),
        f"predicted_{target_col}": y_pred.round(4),
        "absolute_error":          np.abs(y_true - y_pred).round(4),
    })

    results_df.to_csv(CONFIG["output_csv"], index=False)
    print(f"    Predictions saved → {CONFIG['output_csv']}")
    print(f"    Rows: {len(results_df)}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  Model 2 — LSTM Flow Duration Predictor (RT-IoT2022)")
    print("  Target  : flow_duration  [time series regression]")
    print("=" * 65, "\n")

    # Step 1: Load raw data
    df = load_data()

    # Step 2: Preprocess — build sliding-window sequences
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     target_scaler, target_col, feat_scaler) = preprocess(df)

    # Step 3: Build LSTM architecture from scratch
    print("[3/6] Building LSTM model …")
    seq_len    = X_train.shape[1]
    n_features = X_train.shape[2]
    model = build_model(seq_len=seq_len, n_features=n_features)

    # Step 4: Train
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Step 5: Evaluate
    results = evaluate_model(model, X_test, y_test, target_scaler, target_col)

    # Step 6: Save predictions
    save_predictions(results["y_true"], results["y_pred"], target_col)

    # Step 7: Save training history
    print("[6/6] Saving training history …")
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("../outputs/model2_training_history.csv", index_label="epoch")
    print("    History saved → model2_training_history.csv")

    print("\n Done. Summary:")
    print(f"    RMSE (log1p) : {results['rmse']:.4f}")
    print(f"    MAE  (log1p) : {results['mae']:.4f}")
    print(f"    R²   (log1p) : {results['r2']:.4f}")
    print(f"    RMSE (s)     : {results['rmse_orig']:.4f}")
    print(f"    MAE  (s)     : {results['mae_orig']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
