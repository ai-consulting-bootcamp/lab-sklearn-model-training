"""
Breast Cancer Prediction with KNN
Author: Rafael Jurado
Description: Explore the breast cancer dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

print("Loading breast cancer dataset...")
cancer_data = load_breast_cancer()

print(f"\nDataset type: {type(cancer_data)}")
print(f"Number of samples: {len(cancer_data.data)}")
print(f"Number of features: {len(cancer_data.feature_names)}")
print(f"Target classes: {cancer_data.target_names}")

df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df["target"] = cancer_data.target

print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
df.info()

print("\nTarget distribution:")
print(df["target"].value_counts())
print(f"Malignant (0): {(df['target'] == 0).sum()}")
print(f"Benign (1): {(df['target'] == 1).sum()}")

print("\n" + "=" * 50)
print("BASIC STATISTICS")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values found!")
else:
    print(missing[missing > 0])

print("\n" + "=" * 50)
print("FEATURE DISTRIBUTIONS")
print("=" * 50)

key_features = ["mean radius", "mean texture", "mean perimeter", "mean area"]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    # En este dataset: target 1 = benigno, target 0 = maligno
    axes[idx].hist(
        df[df["target"] == 1][feature],
        alpha=0.5,
        label="Benign",
        bins=30,
    )
    axes[idx].hist(
        df[df["target"] == 0][feature],
        alpha=0.5,
        label="Malignant",
        bins=30,
    )
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel("Frequency")
    axes[idx].set_title(f"Distribution of {feature}")
    axes[idx].legend()

plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=150, bbox_inches="tight")
print("Saved visualization to 'feature_distributions.png'")
plt.show()

# Separate features (X) from the target (y)
X = df.drop("target", axis=1)
y = df["target"]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\n" + "=" * 50)
print("DATA SPLIT")
print("=" * 50)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Training features: {X_train.shape[1]}")
print(f"Test features: {X_test.shape[1]}")

print("\nTraining set target distribution:")
print(f"Malignant (0): {(y_train == 0).sum()}")
print(f"Benign (1): {(y_train == 1).sum()}")

print("\nTest set target distribution:")
print(f"Malignant (0): {(y_test == 0).sum()}")
print(f"Benign (1): {(y_test == 1).sum()}")

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("\nKNN classifier trained successfully!")
print(f"Number of neighbors (k): {knn.n_neighbors}")

# Make predictions on both sets
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

print(f"\nTraining predictions: {len(y_train_pred)}")
print(f"Test predictions: {len(y_test_pred)}")

comparison = pd.DataFrame({
    "Actual (0=malignant, 1=benign)": y_test.values,
    "Predicted (0=malignant, 1=benign)": y_test_pred,
})

print("\nFirst 10 actual vs predicted:")
print(comparison.head(10))

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Para esta evaluación, tratamos "malignant (0)" como la clase positiva.
test_precision = precision_score(y_test, y_test_pred, pos_label=0)
test_recall = recall_score(y_test, y_test_pred, pos_label=0)

test_confusion = confusion_matrix(
    y_test,
    y_test_pred,
    labels=[0, 1],
)

print("\n=== Model Performance ===")
print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"Test precision for malignant cases: {test_precision:.4f}")
print(f"Test recall for malignant cases: {test_recall:.4f}")

print("\n=== Confusion Matrix ===")
print("Rows = actual; columns = predicted")
print("                 Predicted")
print("                 Malignant (0)  Benign (1)")
print(
    f"Actual malignant (0)  "
    f"{test_confusion[0, 0]:13d}  {test_confusion[0, 1]:10d}"
)
print(
    f"Actual benign (1)     "
    f"{test_confusion[1, 0]:13d}  {test_confusion[1, 1]:10d}"
)

print("\n=== Classification Report ===")
print(
    classification_report(
        y_test,
        y_test_pred,
        labels=[0, 1],
        target_names=["Malignant (0)", "Benign (1)"],
    )
)

print("\n" + "=" * 50)
print("EXPERIMENTING WITH DIFFERENT K VALUES")
print("=" * 50)

k_values = [1, 3, 5, 7, 9, 11]
results = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)

    y_pred_temp = knn_temp.predict(X_test)

    acc = accuracy_score(y_test, y_pred_temp)
    # Mantener la clase maligna (0) como positiva en estas métricas
    prec = precision_score(y_test, y_pred_temp, pos_label=0)
    rec = recall_score(y_test, y_pred_temp, pos_label=0)

    results.append({
        "K": k,
        "Accuracy": acc,
        "Precision malignant": prec,
        "Recall malignant": rec,
    })

    print(
        f"K={k:2d}: Accuracy={acc:.4f}, "
        f"Precision malignant={prec:.4f}, "
        f"Recall malignant={rec:.4f}"
    )

results_df = pd.DataFrame(results)
best_result = results_df.loc[results_df["Accuracy"].idxmax()]
best_k = int(best_result["K"])

print(
    f"\nBest K by accuracy: {best_k} "
    f"(Accuracy: {best_result['Accuracy']:.4f})"
)

plt.figure(figsize=(10, 6))
plt.plot(results_df["K"], results_df["Accuracy"], marker="o", label="Accuracy")
plt.plot(
    results_df["K"],
    results_df["Precision malignant"],
    marker="s",
    label="Precision malignant",
)
plt.plot(
    results_df["K"],
    results_df["Recall malignant"],
    marker="^",
    label="Recall malignant",
)
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Score")
plt.title("KNN Performance vs K Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("knn_k_comparison.png", dpi=150, bbox_inches="tight")
print("Saved visualization to 'knn_k_comparison.png'")
plt.show()