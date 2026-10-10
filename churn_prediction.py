
# LAB Part 2: Customer Churn Prediction
# Using K-Nearest Neighbors (KNN)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# STEP 1: Load and explore data
import kagglehub
import os

path = kagglehub.dataset_download("blastchar/telco-customer-churn")
df = pd.read_csv(os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv"))

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nStatistics:\n", df.describe(include="all"))
print("\nMissing values:\n", df.isnull().sum())
print("\nChurn distribution:\n", df["Churn"].value_counts())

df["Churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig("churn_distribution.png")
plt.close()

# STEP 2: Preprocess data
df = df.drop(columns=["customerID"])

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Fill missing numeric values with median
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Convert target to 0 and 1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Convert categorical features into numeric columns
df = pd.get_dummies(df, drop_first=True)

print("\nProcessed dataset:", df.shape)

# STEP 3: Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# STEP 4: Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("\nChurn model trained successfully!")

# STEP 5: Predict and evaluate
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# STEP 6: Experiment with K values
results = []

for k in [1, 3, 5, 7, 9, 11, 15]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    results.append({
        "K": k,
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction),
        "Recall": recall_score(y_test, prediction)
    })

results_df = pd.DataFrame(results)

print("\nK Comparison:\n", results_df)

results_df.plot(x="K", y=["Accuracy", "Precision", "Recall"],
                marker="o")
plt.title("Churn Model Performance")
plt.ylabel("Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("churn_k_comparison.png")
plt.close()

best_k = results_df.loc[
    results_df["Accuracy"].idxmax(), "K"
]

print("\nBest K by test accuracy:", best_k)

# STEP 7: Analysis and recommendations
print("\nCONCLUSIONS")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("Best K in experiment:", best_k)

print("\nBusiness recommendations:")
print("- Identify customers at risk of leaving.")
print("- Offer targeted retention incentives.")
print("- Investigate churn patterns by contract and tenure.")

print("\nLimitations:")
print("- KNN is sensitive to feature scaling.")
print("- Customer behavior can change over time.")
print("- Predictions are not always correct.")
print("- K should be selected using cross-validation.")

# Save brief report
with open("churn_report.txt", "w") as report:
    report.write("CUSTOMER CHURN PREDICTION REPORT\n\n")
    report.write("Approach:\n")
    report.write("Loaded and cleaned customer data, encoded categorical "
                 "features, scaled measurements, and trained KNN.\n\n")
    report.write(f"Accuracy: {accuracy:.2%}\n")
    report.write(f"Precision: {precision:.2%}\n")
    report.write(f"Recall: {recall:.2%}\n")
    report.write(f"Best K in test comparison: {best_k}\n\n")
    report.write("Findings:\n")
    report.write("The model identifies patterns associated with churn. "
                 "Contract type, tenure, and monthly charges should be "
                 "investigated as possible churn indicators.\n\n")
    report.write("Recommendations:\n")
    report.write("Use predictions to prioritize customer retention outreach "
                 "and evaluate whether interventions reduce churn.\n\n")
    report.write("Limitations:\n")
    report.write("KNN depends on scaling and may make incorrect predictions. "
                 "Feature importance has not been established. "
                 "Future improvements include cross-validation, "
                 "class imbalance analysis, and model comparisons.\n")
