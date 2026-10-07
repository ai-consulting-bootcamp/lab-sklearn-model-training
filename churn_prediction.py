import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

print("\nData types and non-null counts:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nChurn distribution:")
print(df["Churn"].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for churn_status, color in [("No", "blue"), ("Yes", "orange")]:
    group = df[df["Churn"] == churn_status]
    axes[0].hist(group["tenure"], alpha=0.5, label=churn_status, color=color)
    axes[1].hist(group["MonthlyCharges"], alpha=0.5, label=churn_status, color=color)

axes[0].set_title("Tenure by Churn")
axes[0].set_xlabel("Tenure (months)")
axes[0].set_ylabel("Number of customers")
axes[0].legend(title="Churn")

axes[1].set_title("Monthly Charges by Churn")
axes[1].set_xlabel("Monthly charges")
axes[1].set_ylabel("Number of customers")
axes[1].legend(title="Churn")

plt.tight_layout()
plt.show()

print(df.loc[df["TotalCharges"].isnull(), ["tenure", "MonthlyCharges"]])

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("Missing TotalCharges after conversion:", df["TotalCharges"].isnull().sum())
print(df.loc[df["TotalCharges"].isnull(), ["tenure", "MonthlyCharges", "TotalCharges"]])

print("Missing TotalCharges:", df["TotalCharges"].isnull().sum())
print(df.loc[df["TotalCharges"].isnull(), ["tenure", "MonthlyCharges"]])

df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("Missing values remaining:", df.isnull().sum().sum())
print("TotalCharges type:", df["TotalCharges"].dtype)

df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

df = pd.get_dummies(df, drop_first=True, dtype=int)

print("Shape after encoding:", df.shape)
print("Remaining text columns:", df.select_dtypes(include="object").columns.tolist())
print(df.head())

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Churn precision:", precision_score(y_test, y_pred))
print("Churn recall:", recall_score(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(
    y_test, y_pred, target_names=["No Churn", "Churn"]
))

print("Model trained!")

print("X shape:", X.shape)
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)
print("Churn counts in y:")
print(y.value_counts())

k_values = [1, 3, 5, 7, 9, 11, 15]
results = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    y_pred_temp = knn_temp.predict(X_test)

    acc = accuracy_score(y_test, y_pred_temp)
    prec = precision_score(y_test, y_pred_temp)
    rec = recall_score(y_test, y_pred_temp)

    results.append({
        "K": k,
        "Accuracy": acc,
        "Churn Precision": prec,
        "Churn Recall": rec
    })

    print(
        f"K={k}: Accuracy={acc:.4f}, "
        f"Churn Precision={prec:.4f}, Churn Recall={rec:.4f}"
    )

scaled_results = []

for k in k_values:
    knn_scaled = KNeighborsClassifier(n_neighbors=k)
    knn_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = knn_scaled.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred_scaled)
    prec = precision_score(y_test, y_pred_scaled)
    rec = recall_score(y_test, y_pred_scaled)

    scaled_results.append({
        "K": k,
        "Accuracy": acc,
        "Churn Precision": prec,
        "Churn Recall": rec
    })

    print(
        f"Scaled K={k}: Accuracy={acc:.4f}, "
        f"Churn Precision={prec:.4f}, Churn Recall={rec:.4f}"
    )

    final_knn = KNeighborsClassifier(n_neighbors=11)
final_knn.fit(X_train_scaled, y_train)

final_pred = final_knn.predict(X_test_scaled)

print("Final accuracy:", accuracy_score(y_test, final_pred))
print("Final churn precision:", precision_score(y_test, final_pred))
print("Final churn recall:", recall_score(y_test, final_pred))
print("Final confusion matrix:")
print(confusion_matrix(y_test, final_pred))
print(classification_report(
    y_test, final_pred, target_names=["No Churn", "Churn"]
))