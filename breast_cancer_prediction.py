import pandas as pd
from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

cancer_data = load_breast_cancer()

print(type(cancer_data))
print(cancer_data.data.shape)
print(cancer_data.feature_names[:5])
print(cancer_data.target_names)

df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df["target"] = cancer_data.target

print(df.head())
print(df.info())
print(df["target"].value_counts())

print(df.describe())

missing = df.isnull().sum()
print("\nMissing values:")
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found")

key_features = ["mean radius", "mean texture", "mean perimeter", "mean area"]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    axes[idx].hist(df[df["target"] == 0][feature], alpha=0.5, label="Malignant", bins=30)
    axes[idx].hist(df[df["target"] == 1][feature], alpha=0.5, label="Benign", bins=30)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel("Frequency")
    axes[idx].set_title(f"Distribution of {feature}")
    axes[idx].legend()

plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=cancer_data.target_names))

print("First 10 predictions:", y_pred[:10])
print("First 10 actual labels:", y_test.iloc[:10].to_list())

print("Model trained!")

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

k_values = [1, 3, 5, 7, 9, 11]
results = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)

    y_pred_temp = knn_temp.predict(X_test)

    acc = accuracy_score(y_test, y_pred_temp)
    prec_malignant = precision_score(y_test, y_pred_temp, pos_label=0)
    rec_malignant = recall_score(y_test, y_pred_temp, pos_label=0)

    results.append({
        "K": k,
        "Accuracy": acc,
        "Malignant Precision": prec_malignant,
        "Malignant Recall": rec_malignant
    })

    print(
        f"K={k}: Accuracy={acc:.4f}, "
        f"Malignant Precision={prec_malignant:.4f}, "
        f"Malignant Recall={rec_malignant:.4f}"
    )

results_df = pd.DataFrame(results)
print(results_df)

best_k = results_df.loc[results_df["Accuracy"].idxmax(), "K"]
print(f"\nBest K value: {best_k}")

plt.figure(figsize=(10, 6))
plt.plot(results_df["K"], results_df["Accuracy"], marker="o", label="Accuracy")
plt.plot(
    results_df["K"],
    results_df["Malignant Precision"],
    marker="s",
    label="Malignant Precision"
)
plt.plot(
    results_df["K"],
    results_df["Malignant Recall"],
    marker="^",
    label="Malignant Recall"
)
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Score")
plt.title("KNN Performance vs K Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("knn_k_comparison.png", dpi=150, bbox_inches="tight")
plt.show()