
from sklearn.ensemble import IsolationForest

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

import joblib
import pandas as pd

# Column names for the NSL-KDD dataset
columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty"
]

# Load the dataset
data = pd.read_csv(
    "data/KDDTrain+.txt",
    header=None,
    names=columns
)

# Display the first 5 rows
print("Number of rows and columns:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nDataset information:")
print(data.info())
print("\nLabels in the dataset:")
print(data["label"].value_counts())

# Convert the original labels into two categories
data["binary_label"] = data["label"].apply(
    lambda label: "Normal" if label == "normal" else "Attack"
)

print("\nNormal vs Attack distribution:")
print(data["binary_label"].value_counts())

print("\nFirst 10 original and converted labels:")
print(data[["label", "binary_label"]].head(10))

print("\nColumns containing text:")
print(data.select_dtypes(include=["str", "object"]).columns.tolist())

print("\nUnique values in protocol_type:")
print(data["protocol_type"].unique())

print("\nNumber of unique services:")
print(data["service"].nunique())

print("\nUnique values in flag:")
print(data["flag"].unique())

# Step 8: Test one-hot encoding

categorical_columns = ["protocol_type", "service", "flag"]

encoded_sample = pd.get_dummies(
    data[categorical_columns].head(5)
)

print("\nOriginal categorical data:")
print(data[categorical_columns].head(5))

print("\nData after one-hot encoding:")
print(encoded_sample)

print("\nShape after encoding:")
print(encoded_sample.shape)

# Step 9: Check numerical and categorical features

print("\nDataset shape:")
print(data.shape)

print("\nNumber of unique values in each categorical feature:")
for column in ["protocol_type", "service", "flag"]:
    print(column, ":", data[column].nunique())

print("\nNumber of duplicate rows:")
print(data.duplicated().sum())

print("\nMissing values in the dataset:")
print(data.isnull().sum().sum())

# Step 10: Separate features and target

# Remove label columns from the input features
X = data.drop(columns=["label", "binary_label", "difficulty"])

# Store the answer separately
y = data["binary_label"]

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nTarget distribution:")
print(y.value_counts())
print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Step 11: Prepare categorical and numerical features

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_columns = ["protocol_type", "service", "flag"]

numerical_columns = [
    column for column in X.columns
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
            "passthrough",
            numerical_columns
        )
    ]
)

print("\nCategorical features:", len(categorical_columns))
print("Numerical features:", len(numerical_columns))
print("Preprocessor created successfully!")

# Step 12: Split the training dataset

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining features:", X_train.shape)
print("Validation features:", X_val.shape)

print("\nTraining labels:")
print(y_train.value_counts())

print("\nValidation labels:")
print(y_val.value_counts())

# Step 13: Train the Random Forest model

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed successfully!")

# Step 14: Evaluate the Random Forest model

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

print("\nEvaluating Random Forest model...")

# Predict labels for the validation dataset
y_pred = model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Detailed performance report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred, labels=["Normal", "Attack"])
print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Attack"]
)
display.plot()
plt.title("Random Forest - Validation Results")
plt.tight_layout()
plt.show()


# Step 15: Load the separate NSL-KDD test dataset
print("\nLoading the separate test dataset...")

test_data = pd.read_csv(
    "data/KDDTest+.txt",
    header=None,
    names=columns
)

print("Test dataset shape:", test_data.shape)

print("\nFirst five test records:")
print(test_data.head())

print("\nTest labels:")
print(test_data["label"].value_counts().head(10))

# Step 16: Prepare test data and evaluate Random Forest

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Convert original test labels into Normal or Attack
test_data["binary_label"] = test_data["label"].apply(
    lambda label: "Normal" if label == "normal" else "Attack"
)

# Separate test features and true labels
X_test = test_data.drop(
    columns=["label", "binary_label", "difficulty"]
)
y_test = test_data["binary_label"]

# Predict using the already-trained model
print("\nEvaluating on the separate KDDTest+ dataset...")
y_test_pred = model.predict(X_test)

print(
    f"\nTest Accuracy: "
    f"{accuracy_score(y_test, y_test_pred) * 100:.2f}%"
)

print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred, labels=["Normal", "Attack"]))


# Step 17: Investigate train-test differences

print("\nFeature columns match:")
print(list(X.columns) == list(X_test.columns))

train_attack_types = set(data.loc[data["label"] != "normal", "label"])
test_attack_types = set(test_data.loc[test_data["label"] != "normal", "label"])

unseen_attack_types = test_attack_types - train_attack_types

print("\nAttack types in training data:", len(train_attack_types))
print("Attack types in test data:", len(test_attack_types))
print("Attack types absent from training:", len(unseen_attack_types))

print("\nExamples of unseen attack types:")
print(sorted(unseen_attack_types)[:20])

# Step 19: Count unseen attack types in the test dataset

unseen_attack_data = test_data[
    test_data["label"].isin(unseen_attack_types)
]

print("\nNumber of test records belonging to unseen attack types:")
print(len(unseen_attack_data))

print("\nUnseen attack types and their record counts:")
print(unseen_attack_data["label"].value_counts())

print("\nPercentage of the test dataset with unseen attack types:")
print(
    round(
        len(unseen_attack_data) / len(test_data) * 100,
        2
    ),
    "%"
)

# Step 21: Evaluate detection of unseen attack types

unseen_mask = test_data["label"].isin(unseen_attack_types)

unseen_actual = y_test[unseen_mask]
unseen_predicted = y_test_pred[unseen_mask]

print("\n--- Unseen Attack Detection Results ---")

print("Total unseen attack records:", len(unseen_actual))

print(
    "Correctly detected as Attack:",
    (unseen_predicted == "Attack").sum()
)

print(
    "Incorrectly classified as Normal:",
    (unseen_predicted == "Normal").sum()
)

print(
    "Unseen attack detection rate:",
    round(
        (unseen_predicted == "Attack").mean() * 100,
        2
    ),
    "%"
)
# Step 24: Save the trained Random Forest model

joblib.dump(model, "random_forest_model.pkl")

print("\nRandom Forest model saved successfully!")
# Step 25: Verify the saved model

loaded_model = joblib.load("random_forest_model.pkl")

sample_data = X_test.head(5)

sample_predictions = loaded_model.predict(sample_data)

print("\n--- Saved Model Verification ---")
print("Model loaded successfully!")
print("Predictions for 5 test records:")
print(sample_predictions)

# Step 27: Prepare training data for Isolation Forest

# Select only normal records from the training split
X_if_train = X_train[y_train == "Normal"]

print("\n--- Isolation Forest: Data Preparation ---")
print("Normal training records:", len(X_if_train))
print("Number of input features:", X_if_train.shape[1])


# Step 29: Preprocess data for Isolation Forest

# Create a separate preprocessor for Isolation Forest
from sklearn.base import clone

if_preprocessor = clone(preprocessor)

# Learn the categorical encoding from normal training data only
X_if_train_processed = if_preprocessor.fit_transform(X_if_train)

print("\n--- Isolation Forest: Preprocessing ---")
print("Original features:", X_if_train.shape[1])
print("Processed features:", X_if_train_processed.shape[1])
print("Processed data is numerical and ready for training.")

# Step 30: Train the Isolation Forest model

isolation_forest = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

isolation_forest.fit(X_if_train_processed)

print("\nIsolation Forest trained successfully!")

# Step 32: Evaluate Isolation Forest on validation data

# Transform validation data using the fitted preprocessor
X_val_processed = if_preprocessor.transform(X_val)

# Isolation Forest returns:
#  1  = normal
# -1  = anomaly
if_val_predictions = isolation_forest.predict(X_val_processed)

# Convert predictions to our project labels
if_val_labels = [
    "Normal" if prediction == 1 else "Attack"
    for prediction in if_val_predictions
]

print("\n--- Isolation Forest Validation Results ---")

print("Validation records:", len(X_val))
print("\nClassification Report:")

print(
    classification_report(
        y_val,
        if_val_labels,
        labels=["Attack", "Normal"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_val,
        if_val_labels,
        labels=["Normal", "Attack"]
    )
)


# Step 35: Combine and save the complete Isolation Forest pipeline

isolation_forest_pipeline = Pipeline([
    ("preprocessor", if_preprocessor),
    ("detector", isolation_forest)
])

# Save both components in a single file
joblib.dump(
    isolation_forest_pipeline,
    "isolation_forest_model.pkl"
)

print("\nIsolation Forest pipeline saved successfully!")


# Step 36: Verify the saved pipeline

loaded_if_pipeline = joblib.load(
    "isolation_forest_model.pkl"
)

sample_data = X_test.head(5)

sample_predictions = loaded_if_pipeline.predict(sample_data)

print("\n--- Isolation Forest Pipeline Verification ---")
print("Pipeline loaded successfully!")
print("Predictions for 5 test records:", sample_predictions)
print("Expected prediction values: 1 = Normal, -1 = Anomaly")



# Verify the combined Isolation Forest pipeline

combined_model = joblib.load("isolation_forest_model.pkl")

sample_data = X_test.head(5)
sample_predictions = combined_model.predict(sample_data)

print("\nCombined Isolation Forest pipeline works!")
print("Sample predictions:", sample_predictions)

# Step 37: Evaluate Isolation Forest on the test dataset

# Predict using the combined pipeline
if_test_predictions = loaded_if_pipeline.predict(X_test)

# Convert Isolation Forest output to project labels
if_test_labels = [
    "Normal" if prediction == 1 else "Attack"
    for prediction in if_test_predictions
]

print("\n--- Isolation Forest Test Results ---")

print(
    "Test Accuracy:",
    f"{accuracy_score(y_test, if_test_labels) * 100:.2f}%"
)

print("\nTest Classification Report:")
print(
    classification_report(
        y_test,
        if_test_labels,
        labels=["Attack", "Normal"],
        zero_division=0
    )
)

print("\nTest Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        if_test_labels,
        labels=["Normal", "Attack"]
    )
)

# Step 40: Evaluate Isolation Forest on unseen attack types

unseen_mask = test_data["label"].isin(unseen_attack_types)

unseen_if_predictions = [
    if_test_labels[i]
    for i in range(len(test_data))
    if unseen_mask.iloc[i]
]

print("\n--- Isolation Forest: Unseen Attack Results ---")

print("Unseen attack records:", len(unseen_if_predictions))

print(
    "Detected as Attack:",
    sum(label == "Attack" for label in unseen_if_predictions)
)

print(
    "Missed as Normal:",
    sum(label == "Normal" for label in unseen_if_predictions)
)

print(
    "Unseen attack detection rate:",
    f"{sum(label == 'Attack' for label in unseen_if_predictions) / len(unseen_if_predictions) * 100:.2f}%"
)

# Step 44: Prepare normal training data for One-Class SVM

X_svm_train = X_train[y_train == "Normal"]

print("\n--- One-Class SVM: Data Preparation ---")
print("Normal training records:", len(X_svm_train))
print("Original input features:", X_svm_train.shape[1])

# Step 45: Preprocess and scale data for One-Class SVM

from sklearn.base import clone

# Create a separate preprocessor for One-Class SVM
svm_preprocessor = clone(preprocessor)

# Learn the encoding using normal training records only
X_svm_train_encoded = svm_preprocessor.fit_transform(X_svm_train)

# Convert the encoded data to a dense array for scaling
X_svm_train_encoded = X_svm_train_encoded.toarray() if hasattr(
    X_svm_train_encoded, "toarray"
) else X_svm_train_encoded

# Create and fit the scaler on training data only
svm_scaler = StandardScaler()

X_svm_train_scaled = svm_scaler.fit_transform(
    X_svm_train_encoded
)

print("\n--- One-Class SVM: Preprocessing Results ---")
print("Features after encoding:", X_svm_train_scaled.shape[1])
print("Training records:", X_svm_train_scaled.shape[0])
print("Scaling completed successfully!")

# Step 46: Train One-Class SVM on a manageable sample

import numpy as np
from sklearn.svm import OneClassSVM

# Select 5,000 normal records for training
rng = np.random.default_rng(42)
sample_indices = rng.choice(
    len(X_svm_train_scaled),
    size=min(5000, len(X_svm_train_scaled)),
    replace=False
)

X_svm_sample = X_svm_train_scaled[sample_indices]

print("\n--- One-Class SVM: Training ---")
print("Records selected:", X_svm_sample.shape[0])
print("Features used:", X_svm_sample.shape[1])

# Train the anomaly detection model
one_class_svm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05
)

one_class_svm.fit(X_svm_sample)

print("One-Class SVM training completed successfully!")

# Step 47: Evaluate One-Class SVM on validation data

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Transform validation data using the existing fitted preprocessor
X_svm_val_encoded = svm_preprocessor.transform(X_val)

# Convert sparse data to a dense array if necessary
if hasattr(X_svm_val_encoded, "toarray"):
    X_svm_val_encoded = X_svm_val_encoded.toarray()

# Apply the same scaler used during training
X_svm_val_scaled = svm_scaler.transform(X_svm_val_encoded)

# Predict: 1 = Normal, -1 = Attack
svm_val_predictions = one_class_svm.predict(X_svm_val_scaled)

svm_val_labels = [
    "Normal" if prediction == 1 else "Attack"
    for prediction in svm_val_predictions
]

print("\n--- One-Class SVM: Validation Results ---")
print("Accuracy:", round(accuracy_score(y_val, svm_val_labels) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_val, svm_val_labels, zero_division=0))

print("Confusion Matrix (Normal, Attack):")
print(confusion_matrix(y_val, svm_val_labels, labels=["Normal", "Attack"]))

# Step 50: Evaluate One-Class SVM on the separate test dataset

# Preprocess test records using the fitted preprocessor
X_svm_test_encoded = svm_preprocessor.transform(X_test)

# Convert sparse matrix to dense array if necessary
if hasattr(X_svm_test_encoded, "toarray"):
    X_svm_test_encoded = X_svm_test_encoded.toarray()

# Apply the already-fitted scaler
X_svm_test_scaled = svm_scaler.transform(X_svm_test_encoded)

# Predict: 1 = Normal, -1 = Attack
svm_test_predictions = one_class_svm.predict(X_svm_test_scaled)

svm_test_labels = [
    "Normal" if prediction == 1 else "Attack"
    for prediction in svm_test_predictions
]

print("\n--- One-Class SVM: Test Results ---")

print(
    "Test Accuracy:",
    round(accuracy_score(y_test, svm_test_labels) * 100, 2),
    "%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        svm_test_labels,
        zero_division=0
    )
)

print("Confusion Matrix (Normal, Attack):")
print(
    confusion_matrix(
        y_test,
        svm_test_labels,
        labels=["Normal", "Attack"]
    )
)

# Step 53: Save the complete One-Class SVM pipeline

import joblib
from sklearn.pipeline import Pipeline

one_class_svm_pipeline = Pipeline([
    ("preprocessor", svm_preprocessor),
    ("scaler", svm_scaler),
    ("detector", one_class_svm)
])

joblib.dump(one_class_svm_pipeline, "one_class_svm_model.pkl")

print("\nOne-Class SVM pipeline saved successfully!")
print("File: one_class_svm_model.pkl")
