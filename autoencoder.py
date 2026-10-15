
# Step 71: Load the NSL-KDD training dataset

import pandas as pd
from pathlib import Path
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

train_path = Path("data") / "KDDTrain+.txt"

data = pd.read_csv(
    train_path,
    header=None,
    names=columns
)

# Identify normal traffic
normal_data = data[data["label"] == "normal"].copy()

# Keep the 41 input features; remove label and difficulty
X_normal = normal_data.drop(columns=["label", "difficulty"])

print("--- Autoencoder: Dataset Loaded ---")
print("Total training records:", len(data))
print("Normal records:", len(X_normal))
print("Input features:", X_normal.shape[1])
print("Dataset loaded successfully!")

# Step 73: Encode categorical features and scale numerical features

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_columns = ["protocol_type", "service", "flag"]

numerical_columns = [
    column for column in X_normal.columns
    if column not in categorical_columns
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)

# Learn preprocessing rules from normal training records
X_normal_processed = preprocessor.fit_transform(X_normal)

# Convert the result to a dense array for TensorFlow
if hasattr(X_normal_processed, "toarray"):
    X_normal_processed = X_normal_processed.toarray()

print("\n--- Autoencoder: Preprocessing Results ---")
print("Normal training records:", X_normal_processed.shape[0])
print("Features after encoding:", X_normal_processed.shape[1])
print("Numeric conversion successful:", X_normal_processed.dtype)
print("Preprocessing completed successfully!")

# Step 74: Split normal traffic into training and validation sets

from sklearn.model_selection import train_test_split

X_ae_train, X_ae_val = train_test_split(
    X_normal_processed,
    test_size=0.20,
    random_state=42
)

print("\n--- Autoencoder: Data Split ---")
print("Training records:", X_ae_train.shape[0])
print("Validation records:", X_ae_val.shape[0])
print("Features per record:", X_ae_train.shape[1])
print("Data split completed successfully!")
# Step 75: Build the Autoencoder neural network
import tensorflow as tf
from tensorflow.keras import layers, Model

input_dim = X_ae_train.shape[1]

input_layer = layers.Input(shape=(input_dim,))

# Encoder: compress the input
encoded = layers.Dense(32, activation="relu")(input_layer)
encoded = layers.Dense(16, activation="relu")(encoded)

# Decoder: reconstruct the original input
decoded = layers.Dense(32, activation="relu")(encoded)
output_layer = layers.Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

print("\n--- Autoencoder: Model Created ---")
autoencoder.summary()
# Step 76: Train the Autoencoder
print("\n--- Autoencoder: Training Started ---")

history = autoencoder.fit(
    X_ae_train,
    X_ae_train,
    validation_data=(X_ae_val, X_ae_val),
    epochs=10,
    batch_size=256,
    shuffle=True,
    verbose=1
)

print("\nAutoencoder training completed successfully!")
# Step 77: Calculate reconstruction errors
import numpy as np

X_ae_val_reconstructed = autoencoder.predict(
    X_ae_val,
    verbose=1
)

val_reconstruction_errors = np.mean(
    np.square(X_ae_val - X_ae_val_reconstructed),
    axis=1
)

print("\n--- Autoencoder: Reconstruction Errors ---")
print("Number of validation records:", len(val_reconstruction_errors))
print("Mean error:", np.mean(val_reconstruction_errors))
print("Median error:", np.median(val_reconstruction_errors))
print("95th percentile:", np.percentile(val_reconstruction_errors, 95))
print("99th percentile:", np.percentile(val_reconstruction_errors, 99))
print("Reconstruction error calculation completed!")
# Step 78: Set an initial anomaly threshold
anomaly_threshold = np.percentile(
    val_reconstruction_errors,
    99
)

print("\n--- Autoencoder: Anomaly Threshold ---")
print("Selected percentile: 99th")
print("Anomaly threshold:", anomaly_threshold)

normal_val_predictions = (
    val_reconstruction_errors > anomaly_threshold
)

print(
    "Normal validation records flagged:",
    np.sum(normal_val_predictions)
)
print(
    "Percentage flagged:",
    np.mean(normal_val_predictions) * 100,
    "%"
)
# Step 79: Load and prepare test data

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

test_path = Path("data") / "KDDTest+.txt"
test_data = pd.read_csv(
    test_path,
    header=None,
    names=columns
)

X_ae_test = test_data.drop(
    columns=["label", "difficulty"]
)

y_ae_test = np.where(
    test_data["label"] == "normal",
    "Normal",
    "Attack"
)

X_ae_test_processed = preprocessor.transform(X_ae_test)

if hasattr(X_ae_test_processed, "toarray"):
    X_ae_test_processed = X_ae_test_processed.toarray()

print("\n--- Autoencoder: Test Data Prepared ---")
print("Test records:", len(X_ae_test))
print("Processed features:", X_ae_test_processed.shape[1])
print("Normal records:", np.sum(y_ae_test == "Normal"))
print("Attack records:", np.sum(y_ae_test == "Attack"))
# Step 80: Predict anomalies in the test dataset

X_ae_test_reconstructed = autoencoder.predict(
    X_ae_test_processed,
    verbose=1
)

test_reconstruction_errors = np.mean(
    np.square(
        X_ae_test_processed - X_ae_test_reconstructed
    ),
    axis=1
)

y_ae_pred = np.where(
    test_reconstruction_errors > anomaly_threshold,
    "Attack",
    "Normal"
)

print("\n--- Autoencoder: Test Predictions ---")
print("Total predictions:", len(y_ae_pred))
print("Predicted Normal:", np.sum(y_ae_pred == "Normal"))
print("Predicted Attack:", np.sum(y_ae_pred == "Attack"))
print("Threshold used:", anomaly_threshold)
print("Test predictions completed successfully!")

# Step 81: Evaluate Autoencoder performance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n--- Autoencoder: Test Performance ---")

accuracy = accuracy_score(y_ae_test, y_ae_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_ae_test,
        y_ae_pred,
        labels=["Normal", "Attack"],
        zero_division=0
    )
)

cm = confusion_matrix(
    y_ae_test,
    y_ae_pred,
    labels=["Normal", "Attack"]
)

print("\nConfusion Matrix (Actual rows, Predicted columns):")
print("                 Predicted Normal  Predicted Attack")
print(f"Actual Normal          {cm[0, 0]:6d}           {cm[0, 1]:6d}")
print(f"Actual Attack          {cm[1, 0]:6d}           {cm[1, 1]:6d}")


# Step 83: Save the complete Autoencoder pipeline in one file
import joblib

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

final_model = {
    "autoencoder": autoencoder,
    "preprocessor": preprocessor,
    "anomaly_threshold": float(anomaly_threshold)
}

joblib.dump(final_model, models_dir / "autoencoder_final.pkl")

print("\n--- Complete Autoencoder Saved ---")
print("File: models/autoencoder_final.pkl")
