"""
model1_attack_classifier.py
============================
Model 1 — Deep Neural Network (DNN) Attack-Type Classifier
Target variable : Attack_type  (13-class multi-class classification)
Dataset         : RT-IoT2022  (UCI ML Repository, ID 942)

NOTE: This model predicts the ATTACK TYPE label — a categorical variable
      representing whether a network flow is normal or a specific attack.
      This is intentionally DIFFERENT from Model 2, which predicts the
      continuous variable `flow_duration` via time series regression.

Architecture (all layers built from scratch using TensorFlow/Keras primitives):
    Input(83)  →  Dense(256) → BatchNorm → ReLU → Dropout(0.4)
               →  Dense(128) → BatchNorm → ReLU → Dropout(0.3)
               →  Dense(64)  → BatchNorm → ReLU → Dropout(0.2)
               →  Dense(13)  → Softmax

Usage:
    pip install ucimlrepo tensorflow scikit-learn pandas numpy
    python model1_attack_classifier.py
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Suppress TensorFlow info/warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Fix random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================================================
# Configuration — centralised hyperparameters for easy instructor review
# =============================================================================
CONFIG = {
    "test_size":      0.20,   # 20 % held out as test set
    "val_size":       0.10,   # 10 % of training used for validation
    "batch_size":     512,    # large batch → stable gradient estimates
    "epochs":         50,     # max epochs; early stopping may terminate sooner
    "patience":       7,      # early stopping patience (no val_loss improvement)
    "lr":             1e-3,   # Adam initial learning rate
    "lr_decay":       0.5,    # ReduceLROnPlateau factor
    "lr_patience":    3,      # epochs before LR is reduced
    "dropout_rates":  [0.4, 0.3, 0.2],   # per hidden layer
    "hidden_units":   [256, 128, 64],     # neurons per hidden layer
    "l2_lambda":      1e-4,   # L2 regularisation on Dense weights
    "output_csv":      "../outputs/model1_predictions.csv",
    "model_save_path": "../models/model1_attack_classifier_saved.keras",
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
        Combined DataFrame of features + target column 'Attack_type'.
    """
    print("[1/6] Loading RT-IoT2022 dataset from UCI repository …")
    from ucimlrepo import fetch_ucirepo

    # fetch_ucirepo returns an object with .data.features and .data.targets
    rt_iot2022 = fetch_ucirepo(id=942)

    X_raw = rt_iot2022.data.features   # pandas DataFrame, shape (123117, 83)
    y_raw = rt_iot2022.data.targets    # pandas DataFrame, shape (123117, 1)

    # Combine into a single DataFrame for unified preprocessing
    df = X_raw.copy()
    df["Attack_type"] = y_raw.values.ravel()

    print(f"    Dataset shape  : {df.shape}")
    print(f"    Attack classes : {df['Attack_type'].nunique()}")
    print(f"    Class counts   :\n{df['Attack_type'].value_counts()}\n")
    return df


# =============================================================================
# 2. PREPROCESSING
# =============================================================================

def preprocess(df: pd.DataFrame):
    """
    Prepare features and labels for the DNN classifier.

    Steps
    -----
    1. Drop non-numeric / identifier columns that carry no predictive signal.
    2. One-hot encode low-cardinality categorical features (proto, service).
    3. Impute any residual NaNs with column medians.
    4. Encode the string target label to integer indices.
    5. Standardise all numeric features (zero mean, unit variance).
    6. Split into train / validation / test sets with stratification.
    7. Compute class weights to handle the severe class imbalance.

    Returns
    -------
    tuple : X_train, X_val, X_test, y_train, y_val, y_test,
            class_weight_dict, label_encoder, scaler
    """
    print("[2/6] Preprocessing …")

    # --- Separate target ---
    target_col = "Attack_type"
    y_raw = df[target_col].copy()
    X_raw = df.drop(columns=[target_col])

    # --- Identify categorical columns (object dtype) ---
    cat_cols = X_raw.select_dtypes(include="object").columns.tolist()
    num_cols = X_raw.select_dtypes(exclude="object").columns.tolist()

    # One-hot encode categoricals (proto, service have ~10 unique values each)
    X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=False)

    # Impute any NaN with column median (robust to outliers)
    X_encoded = X_encoded.fillna(X_encoded.median(numeric_only=True))

    # --- Encode target labels to integers ---
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)          # e.g. 'DOS_SYN_Hping' → 0
    n_classes = len(le.classes_)
    print(f"    Encoded classes ({n_classes}): {list(le.classes_)}")

    # --- Train / test split (stratified to preserve class proportions) ---
    X_tr, X_test, y_tr, y_test = train_test_split(
        X_encoded.values, y_enc,
        test_size=CONFIG["test_size"],
        random_state=SEED,
        stratify=y_enc
    )

    # --- Train / validation split ---
    val_frac = CONFIG["val_size"] / (1.0 - CONFIG["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr,
        test_size=val_frac,
        random_state=SEED,
        stratify=y_tr
    )

    # --- Standardise features (fit on train only, transform all splits) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # --- Class weights (inverse frequency) to penalise dominant classes ---
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(cw))

    print(f"    Train / Val / Test : {X_train.shape[0]} / "
          f"{X_val.shape[0]} / {X_test.shape[0]} samples")
    print(f"    Feature dimension  : {X_train.shape[1]}\n")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            class_weight_dict, le, scaler, n_classes)


# =============================================================================
# 3. MODEL ARCHITECTURE
# =============================================================================

def build_model(input_dim: int, n_classes: int) -> keras.Model:
    """
    Build the DNN classifier entirely from scratch using Keras primitives.

    Architecture rationale
    ----------------------
    - Three hidden layers with decreasing width (256→128→64) form a
      funnel that progressively compresses the 83-feature input into
      a compact representation before classification.
    - BatchNormalization after each Dense layer stabilises training by
      reducing internal covariate shift — especially important with the
      large class imbalance present in this dataset.
    - Dropout provides regularisation and reduces overfitting; rates
      decrease with depth because later layers hold higher-level features.
    - L2 weight regularisation further prevents memorisation of the
      majority-class DOS_SYN_Hping pattern.
    - Softmax output produces a proper probability distribution over the
      13 attack-type classes.

    Parameters
    ----------
    input_dim : int   — number of input features after preprocessing
    n_classes : int   — number of target classes (13 for RT-IoT2022)

    Returns
    -------
    keras.Model (uncompiled)
    """
    # Use the Functional API so the architecture is explicit and inspectable
    inputs = keras.Input(shape=(input_dim,), name="flow_features")

    # --- Hidden Layer 1: Dense(256) + BatchNorm + ReLU + Dropout(0.4) ---
    x = layers.Dense(
        CONFIG["hidden_units"][0],
        kernel_regularizer=regularizers.l2(CONFIG["l2_lambda"]),
        name="dense_1"
    )(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation("relu", name="relu_1")(x)
    x = layers.Dropout(CONFIG["dropout_rates"][0], name="drop_1")(x)

    # --- Hidden Layer 2: Dense(128) + BatchNorm + ReLU + Dropout(0.3) ---
    x = layers.Dense(
        CONFIG["hidden_units"][1],
        kernel_regularizer=regularizers.l2(CONFIG["l2_lambda"]),
        name="dense_2"
    )(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Activation("relu", name="relu_2")(x)
    x = layers.Dropout(CONFIG["dropout_rates"][1], name="drop_2")(x)

    # --- Hidden Layer 3: Dense(64) + BatchNorm + ReLU + Dropout(0.2) ---
    x = layers.Dense(
        CONFIG["hidden_units"][2],
        kernel_regularizer=regularizers.l2(CONFIG["l2_lambda"]),
        name="dense_3"
    )(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Activation("relu", name="relu_3")(x)
    x = layers.Dropout(CONFIG["dropout_rates"][2], name="drop_3")(x)

    # --- Output Layer: Dense(n_classes) + Softmax ---
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name="dnn_attack_classifier")
    return model


# =============================================================================
# 4. TRAINING
# =============================================================================

def train_model(model: keras.Model,
                X_train, y_train,
                X_val, y_val,
                class_weight_dict: dict) -> keras.callbacks.History:
    """
    Compile and train the DNN with Adam optimiser and early stopping.

    Callbacks
    ---------
    EarlyStopping   — halt training when val_loss stops improving (patience=7)
    ReduceLROnPlateau — halve LR after 3 epochs without val_loss improvement
    ModelCheckpoint — save the best weights (by val_loss) during training

    Parameters
    ----------
    model            : compiled keras.Model
    X_train, y_train : training features and integer labels
    X_val, y_val     : validation features and integer labels
    class_weight_dict: per-class weight mapping for imbalance correction

    Returns
    -------
    keras.callbacks.History object (contains per-epoch loss / metrics)
    """
    print("[3/6] Compiling and training model …")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["lr"]),
        loss="sparse_categorical_crossentropy",  # labels are integer indices
        metrics=["accuracy"]
    )

    model.summary()

    callbacks = [
        # Stop if validation loss does not improve for `patience` epochs
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CONFIG["patience"],
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when training plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=CONFIG["lr_decay"],
            patience=CONFIG["lr_patience"],
            min_lr=1e-6,
            verbose=1
        ),
        # Persist the epoch with the lowest validation loss
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
        class_weight=class_weight_dict,   # address class imbalance
        callbacks=callbacks,
        verbose=1
    )

    return history


# =============================================================================
# 5. EVALUATION
# =============================================================================

def evaluate_model(model: keras.Model,
                   X_test, y_test,
                   label_encoder: LabelEncoder) -> dict:
    """
    Evaluate the trained classifier on the held-out test set.

    Metrics reported
    ----------------
    - Overall accuracy
    - Macro-averaged F1 (treats all classes equally — important given imbalance)
    - Weighted F1 (accounts for support of each class)
    - Per-class precision / recall / F1 via classification_report

    Parameters
    ----------
    model         : trained keras.Model
    X_test, y_test: held-out test features and integer labels
    label_encoder : fitted LabelEncoder used to recover class names

    Returns
    -------
    dict with scalar metric values
    """
    print("[4/6] Evaluating on test set …")

    # Get raw probability predictions then take argmax for class labels
    y_prob  = model.predict(X_test, batch_size=CONFIG["batch_size"], verbose=0)
    y_pred  = np.argmax(y_prob, axis=1)

    acc        = accuracy_score(y_test, y_pred)
    f1_macro   = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_weighted= f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n    Test Accuracy         : {acc:.4f}")
    print(f"    Macro F1-Score        : {f1_macro:.4f}")
    print(f"    Weighted F1-Score     : {f1_weighted:.4f}")
    print("\n    Per-class Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    return {
        "accuracy":    acc,
        "f1_macro":    f1_macro,
        "f1_weighted": f1_weighted,
        "y_pred":      y_pred,
        "y_prob":      y_prob,
    }


# =============================================================================
# 6. SAVE PREDICTIONS TO CSV
# =============================================================================

def save_predictions(X_test, y_test, y_pred, y_prob,
                     label_encoder: LabelEncoder) -> None:
    """
    Write test-set predictions to a CSV file for dashboard / further analysis.

    Output columns
    --------------
    true_label      : ground-truth class name
    predicted_label : model's predicted class name
    confidence      : softmax probability for the predicted class
    correct         : boolean flag (True if prediction matches ground truth)

    Parameters
    ----------
    X_test        : test feature array (not written to CSV — only used for index)
    y_test        : integer true labels
    y_pred        : integer predicted labels
    y_prob        : softmax probability matrix (n_samples × n_classes)
    label_encoder : to convert integer indices back to human-readable names
    """
    print("[5/6] Saving predictions …")

    results_df = pd.DataFrame({
        "true_label":      label_encoder.inverse_transform(y_test),
        "predicted_label": label_encoder.inverse_transform(y_pred),
        "confidence":      np.max(y_prob, axis=1).round(4),
        "correct":         (y_test == y_pred),
    })

    results_df.to_csv(CONFIG["output_csv"], index=False)
    print(f"    Predictions saved → {CONFIG['output_csv']}")
    print(f"    Rows: {len(results_df)}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  Model 1 — DNN Attack-Type Classifier (RT-IoT2022)")
    print("  Target  : Attack_type  [13-class classification]")
    print("=" * 65, "\n")

    # Step 1: Load raw data from UCI repository
    df = load_data()

    # Step 2: Preprocess — encode, scale, split, compute class weights
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     class_weight_dict, le, scaler, n_classes) = preprocess(df)

    # Step 3: Build DNN architecture from scratch
    print("[3/6] Building model architecture …")
    model = build_model(input_dim=X_train.shape[1], n_classes=n_classes)

    # Step 4: Train with callbacks
    history = train_model(model, X_train, y_train, X_val, y_val,
                          class_weight_dict)

    # Step 5: Evaluate on test set
    results = evaluate_model(model, X_test, y_test, le)

    # Step 6: Save predictions CSV
    save_predictions(X_test, y_test,
                     results["y_pred"], results["y_prob"], le)

    # Step 7: Persist training history for plotting / reporting
    print("[6/6] Saving training history …")
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("../outputs/model1_training_history.csv", index_label="epoch")
    print("    History saved → model1_training_history.csv")

    print("\n Done. Summary:")
    print(f"    Accuracy  : {results['accuracy']:.4f}")
    print(f"    F1 Macro  : {results['f1_macro']:.4f}")
    print(f"    F1 Wtd.   : {results['f1_weighted']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
