## Imports + Step 1 (load dataset 
# and inspect shape/target counts).

"""
Breast Cancer Prediction with KNN
Author: [Bhabishwor Rana]
Description: Predict whether a breast tumor is malignant or benign using KNN
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the breast cancer dataset
print("Loading breast cancer dataset...")
cancer_data = load_breast_cancer()

# The dataset is a Bunch object with 'data', 'target', 'feature_names', etc.
print(f"\nDataset type: {type(cancer_data)}")
print(f"Number of samples: {len(cancer_data.data)}")
print(f"Number of features: {len(cancer_data.feature_names)}")
print(f"Target classes: {cancer_data.target_names}")

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df["target"] = cancer_data.target

print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nTarget distribution:")
print(df["target"].value_counts())
# In sklearn's load_breast_cancer(): 0 = malignant, 1 = benign
print(f"Malignant (0): {(df['target'] == 0).sum()}")
print(f"Benign (1): {(df['target'] == 1).sum()}")

# Note: sklearn's load_breast_cancer() 
# encodes 0 = malignant and 1 = benign (opposite of 
# the background summary), so I compute precision 
# and recall with pos_label=0 to measure malignant
#  detection directly.

## Step2:(summary statistics, missing-value
#check, and feature_distributions.png). 

# Basic statistics
print("\n" + "=" * 50)
print("BASIC STATISTICS")
print("=" * 50)
print(df.describe())

# Check for missing values
print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values found!")
else:
    print(missing[missing > 0])

# Visualize feature distributions (select a few key features)
print("\n" + "=" * 50)
print("FEATURE DISTRIBUTIONS")
print("=" * 50)

key_features = ["mean radius", "mean texture", "mean perimeter", "mean area"]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    # Remember: in sklearn's load_breast_cancer(), 0 = malignant, 1 = benign
    axes[idx].hist(df[df["target"] == 0][feature], alpha=0.5, label="Malignant (0)", bins=30)
    axes[idx].hist(df[df["target"] == 1][feature], alpha=0.5, label="Benign (1)", bins=30)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel("Frequency")
    axes[idx].set_title(f"Distribution of {feature}")
    axes[idx].legend()

plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=150, bbox_inches="tight")
print("Saved visualization to 'feature_distributions.png'")
plt.show()

#Step 3: Splitting the Data

# Separate features and target
X = df.drop("target", axis=1)  # All columns except 'target'
y = df["target"]  # Target column

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split into training and testing sets
# random_state ensures reproducibility
# stratify=y ensures both sets have similar class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 50)
print("DATA SPLIT")
print("=" * 50)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Training features: {X_train.shape[1]}")
print(f"Test features: {X_test.shape[1]}")

# Verify class distribution in both sets
# Remember: in sklearn's load_breast_cancer(), 0 = malignant, 1 = benign
print("\nTraining set target distribution:")
print(y_train.value_counts())
print(
    f"  Malignant (0): {(y_train == 0).sum()} ({(y_train == 0).mean() * 100:.1f}%)"
)
print(
    f"  Benign (1): {(y_train == 1).sum()} ({(y_train == 1).mean() * 100:.1f}%)"
)

print("\nTest set target distribution:")
print(y_test.value_counts())
print(
    f"  Malignant (0): {(y_test == 0).sum()} ({(y_test == 0).mean() * 100:.1f}%)"
)
print(f"  Benign (1): {(y_test == 1).sum()} ({(y_test == 1).mean() * 100:.1f}%)")

#Part 1 - Steps 4 & 5: Training KNN 
# and Making Predictions

# Create KNN classifier
# n_neighbors=5 means the model will look at the 5 nearest neighbors to make a prediction
print("\n" + "=" * 50)
print("TRAINING KNN MODEL (K=5)")
print("=" * 50)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("KNN classifier trained successfully!")
print(f"Number of neighbors (k): {knn.n_neighbors}")

# Make predictions
print("\n" + "=" * 50)
print("PREDICTIONS ON TRAIN AND TEST SETS")
print("=" * 50)

y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

print(f"Training predictions: {len(y_train_pred)}")
print(f"Test predictions:     {len(y_test_pred)}")

# Compare first 10 test predictions with actual values (0 = malignant, 1 = benign)
sample_df = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_test_pred[:10],
    "Actual_Class": [cancer_data.target_names[i] for i in y_test.values[:10]],
    "Predicted_Class": [cancer_data.target_names[i] for i in y_test_pred[:10]],
})
sample_df["Correct"] = sample_df["Actual"] == sample_df["Predicted"]
print("\nFirst 10 test sample predictions:")
print(sample_df.to_string(index=False))

#Step 6: Evaluating Model Performance

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision_default = precision_score(y_test, y_test_pred)
test_recall_default = recall_score(y_test, y_test_pred)
malignant_precision = precision_score(y_test, y_test_pred, pos_label=0)
malignant_recall = recall_score(y_test, y_test_pred, pos_label=0)
test_confusion = confusion_matrix(y_test, y_test_pred)

print("\n=== Model Performance ===")
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
print(f"Test Accuracy:     {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"Malignant (0) Precision: {malignant_precision:.4f} ({malignant_precision * 100:.2f}%)")
print(f"Malignant (0) Recall:    {malignant_recall:.4f} ({malignant_recall * 100:.2f}%)")
print(f"Benign (1) Precision (default pos_label=1): {test_precision_default:.4f}")
print(f"Benign (1) Recall    (default pos_label=1): {test_recall_default:.4f}")

print("\n=== Confusion Matrix [0 = malignant, 1 = benign] ===")
print("                   Predicted")
print("                 Malig(0)  Benig(1)")
print(f"Actual Malig(0)    {test_confusion[0, 0]:4d}      {test_confusion[0, 1]:4d}")
print(f"Actual Benig(1)    {test_confusion[1, 0]:4d}      {test_confusion[1, 1]:4d}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_test_pred, target_names=cancer_data.target_names))

#Part 1 — Step 7: Experimenting with 
# Different K Values

# Experiment with different K values
print("\n" + "=" * 50)
print("EXPERIMENTING WITH DIFFERENT K VALUES")
print("=" * 50)

k_values = [1, 3, 5, 7, 9, 11]
results = []

for k in k_values:
    # Create and train model
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)

    # Make predictions
    y_pred_temp = knn_temp.predict(X_test)

    # Calculate accuracy, benign (default pos_label=1) and malignant (pos_label=0) metrics
    acc = accuracy_score(y_test, y_pred_temp)
    prec = precision_score(y_test, y_pred_temp)
    rec = recall_score(y_test, y_pred_temp)
    mal_prec = precision_score(y_test, y_pred_temp, pos_label=0)
    mal_rec = recall_score(y_test, y_pred_temp, pos_label=0)

    results.append({
        "K": k,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Malig_Prec": mal_prec,
        "Malig_Rec": mal_rec,
    })

    print(
        f"K={k:2d}: Acc={acc:.4f} | "
        f"Malig(0) Prec={mal_prec:.4f}, Rec={mal_rec:.4f} | "
        f"Benign(1) Prec={prec:.4f}, Rec={rec:.4f}"
    )

# Find best K by test accuracy
results_df = pd.DataFrame(results)
best_k = results_df.loc[results_df["Accuracy"].idxmax(), "K"]
print(f"\nBest K value: {best_k} (Accuracy: {results_df['Accuracy'].max():.4f})")

# Visualize results
plt.figure(figsize=(10, 6))
plt.plot(results_df["K"], results_df["Accuracy"], marker="o", label="Accuracy")
plt.plot(results_df["K"], results_df["Malig_Prec"], marker="s", label="Malignant (0) Precision")
plt.plot(results_df["K"], results_df["Malig_Rec"], marker="^", label="Malignant (0) Recall")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Score")
plt.title("KNN Performance vs K Value (Malignant Focus)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("knn_k_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved visualization to 'knn_k_comparison.png'")
# plt.show()  # Uncomment if you want the pop-up window after the script finishes