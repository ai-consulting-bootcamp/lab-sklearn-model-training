from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).parent / "data"
csv_path = DATA_DIR / "telco_churn.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"Dataset not found: {csv_path}")

df = pd.read_csv(csv_path)

print("Dataset loaded successfully")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nChurn distribution:")
print(df["Churn"].value_counts())
print("\nChurn proportions:")
print(df["Churn"].value_counts(normalize=True) * 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df["Churn"].value_counts().reindex(["No", "Yes"]).plot(
    kind="bar",
    ax=axes[0],
    color=["steelblue", "coral"],
)
axes[0].set_title("Customer Churn Distribution")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Number of Customers")
axes[0].tick_params(axis="x", rotation=0)

churn_by_contract = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index",
).reindex(columns=["No", "Yes"])

churn_by_contract.plot(
    kind="bar",
    stacked=True,
    ax=axes[1],
    color=["steelblue", "coral"],
)
axes[1].set_title("Churn Proportion by Contract")
axes[1].set_xlabel("Contract Type")
axes[1].set_ylabel("Proportion")
axes[1].legend(title="Churn")
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("churn_exploration.png", dpi=150, bbox_inches="tight")
print("\nSaved visualization to 'churn_exploration.png'")
plt.show()

# Prepare a separate copy; keep the original data unchanged
df_model = df.copy()

# TotalCharges contains blank strings, so convert it to numeric
df_model["TotalCharges"] = pd.to_numeric(
    df_model["TotalCharges"],
    errors="coerce",
)

print(
    "\nUnusable TotalCharges values:",
    df_model["TotalCharges"].isna().sum(),
)

# Remove customer ID because it identifies a record, not a customer behavior
df_model = df_model.drop(columns=["customerID"])

# Convert the target to binary: 0 = no churn, 1 = churn
df_model["Churn"] = df_model["Churn"].map({"No": 0, "Yes": 1})

# Fill missing numeric values with the median
numeric_columns = df_model.select_dtypes(include="number").columns
for column in numeric_columns:
    if df_model[column].isna().any():
        df_model[column] = df_model[column].fillna(
            df_model[column].median()
        )

# Fill missing categorical values with the most common value
categorical_columns = df_model.select_dtypes(exclude="number").columns
for column in categorical_columns:
    if df_model[column].isna().any():
        df_model[column] = df_model[column].fillna(
            df_model[column].mode().iloc[0]
        )

# Encode categorical predictors as numeric indicator columns
X = pd.get_dummies(
    df_model.drop(columns=["Churn"]),
    drop_first=True,
    dtype=int,
)
y = df_model["Churn"]

print("\nPreprocessing complete")
print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Missing values after preprocessing:", df_model.isna().sum().sum())
print("\nTarget counts (0 = no churn, 1 = churn):")
print(y.value_counts().sort_index())

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nData split complete")
print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])
print("Training features:", X_train.shape[1])
print("Test features:", X_test.shape[1])
print("\nChurn counts in training set:")
print(y_train.value_counts().sort_index())
print("\nChurn counts in test set:")
print(y_test.value_counts().sort_index())

# Scale features using only the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train_scaled, y_train)

# Predict churn on the test set
y_pred = knn.predict(X_test_scaled)

print("\nKNN model trained with k=11")
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No churn", "Churn"],
    )
)

from sklearn.metrics import precision_score, recall_score

k_values = [1, 3, 5, 7, 9, 11, 15]
results = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    results.append({
        "k": k,
        "accuracy": accuracy_score(y_test, predictions),
        "churn_precision": precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "churn_recall": recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
    })

results_df = pd.DataFrame(results)

print("\nKNN comparison by k:")
print(results_df.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.plot(results_df["k"], results_df["accuracy"], marker="o", label="Accuracy")
plt.plot(
    results_df["k"],
    results_df["churn_precision"],
    marker="s",
    label="Churn precision",
)
plt.plot(
    results_df["k"],
    results_df["churn_recall"],
    marker="^",
    label="Churn recall",
)
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Score")
plt.title("KNN Performance vs k")
plt.xticks(k_values)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("churn_knn_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved visualization to 'churn_knn_comparison.png'")
plt.show()