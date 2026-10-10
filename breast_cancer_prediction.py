
# LAB Part 1: Breast Cancer Prediction
# Using K-Nearest Neighbors (KNN)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# STEP 1: Load and explore data
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nStatistics:\n", df.describe())
print("\nDiagnosis distribution:\n", df["target"].value_counts())
print("\nMissing values:", df.isnull().sum().sum())

# 0 = Malignant, 1 = Benign

# STEP 2: Visualize features
features = ["mean radius", "mean texture",
            "mean perimeter", "mean area"]

df.groupby("target")[features].mean().plot(kind="bar")
plt.title("Average Tumor Measurements")
plt.xlabel("Diagnosis: 0 = Malignant, 1 = Benign")
plt.tight_layout()
plt.savefig("feature_distributions.png")
plt.close()

# STEP 3: Split data
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# STEP 4: Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("\nKNN model trained successfully!")

# STEP 5: Make predictions
y_train_pred = knn.predict(X_train)
y_pred = knn.predict(X_test)

print("\nFirst 10 predictions:", y_pred[:10])
print("Actual diagnoses:", y_test.iloc[:10].values)

# STEP 6: Evaluate model
print("\nTraining accuracy:",
      accuracy_score(y_train, y_train_pred))

print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Cancer precision:",
      precision_score(y_test, y_pred, pos_label=0))
print("Cancer recall:",
      recall_score(y_test, y_pred, pos_label=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=["Malignant", "Benign"]
))

# STEP 7: Compare different K values
results = []

for k in [1, 3, 5, 7, 9, 11]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    results.append({
        "K": k,
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(
            y_test, prediction, pos_label=0
        ),
        "Recall": recall_score(
            y_test, prediction, pos_label=0
        )
    })

results_df = pd.DataFrame(results)
print("\nK Comparison:\n", results_df)

results_df.plot(x="K", y=["Accuracy", "Precision", "Recall"],
                marker="o")
plt.title("KNN Performance Comparison")
plt.ylabel("Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_k_comparison.png")
plt.close()

print("\nBest K by test accuracy:",
      results_df.loc[results_df["Accuracy"].idxmax(), "K"])

# Conclusion:
# KNN can classify tumors using their measurements.
# Feature scaling is important because KNN uses distances.
# Recall is important because missing malignant tumors is risky.
# Choosing K using test accuracy is exploratory only;
# proper model selection should use cross-validation.
