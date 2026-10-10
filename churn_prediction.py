# %%
import pandas as pd

# Step 1: Load the Telco Customer Churn dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Explore the dataset
print("Dataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nChurn distribution:")
print(df["Churn"].value_counts())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum().sum())

# Check TotalCharges
print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)

# Check empty spaces in TotalCharges
print("\nEmpty values in TotalCharges:")
print((df["TotalCharges"].astype(str).str.strip() == "").sum())

# Step 2: Data Preprocessing

# Convert TotalCharges to numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values after conversion
print("\nMissing values after conversion:")
print(df["TotalCharges"].isnull().sum())

# Fill missing values with the median


# Verify the result
print("\nMissing values after cleaning:")
print(df["TotalCharges"].isnull().sum())

print("\nTotalCharges data type after cleaning:")
print(df["TotalCharges"].dtype)

# Remove customer ID because it is not useful for prediction
df = df.drop(columns=["customerID"])

# Convert target variable to binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Convert categorical features to numeric
X = pd.get_dummies(X, dtype=int)

# Verify preprocessing results
print("\nFeatures shape after encoding:")
print(X.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nRemaining categorical columns:")
print(X.select_dtypes(include=["object", "str"]).columns.tolist())

print("\nMissing values in features:")
print(X.isnull().sum().sum())

# Step 3: Split the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set:")
print(X_train.shape)

print("\nTesting set:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))

# Step 4: Feature Scaling

from sklearn.preprocessing import StandardScaler

# Handle missing TotalCharges without data leakage

# Calculate median using training data only
median_total_charges = X_train["TotalCharges"].median()

# Fill missing values in both datasets
X_train["TotalCharges"] = X_train["TotalCharges"].fillna(
    median_total_charges
)

X_test["TotalCharges"] = X_test["TotalCharges"].fillna(
    median_total_charges
)

print("\nMissing values before train-based imputation:")
print("Training:", X_train.isnull().sum().sum())
print("Testing:", X_test.isnull().sum().sum())

# Create the scaler
scaler = StandardScaler()

# Learn scaling parameters from training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to test data
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")

print("\nTraining data shape:")
print(X_train_scaled.shape)

print("\nTesting data shape:")
print(X_test_scaled.shape)

# Step 4: Train KNN Model

from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train_scaled, y_train)

print("\nKNN model trained successfully!")
print("Number of neighbors:", knn.n_neighbors)

# Step 5: Make Predictions and Evaluate

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# Make predictions using the test dataset
y_pred = knn.predict(X_test_scaled)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n" + "=" * 50)
print("KNN MODEL EVALUATION - K = 5")
print("=" * 50)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Experiment with different K values

k_values = [1, 3, 5, 7, 9, 11, 15]

results = []

for k in k_values:

    # Create and train KNN model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    predictions = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    accuracy_k = accuracy_score(y_test, predictions)
    precision_k = precision_score(y_test, predictions)
    recall_k = recall_score(y_test, predictions)

    # Save results
    results.append({
        "K": k,
        "Accuracy": accuracy_k,
        "Precision": precision_k,
        "Recall": recall_k
    })

# Convert results into a DataFrame
results_df = pd.DataFrame(results)

print("\n" + "=" * 50)
print("KNN COMPARISON - DIFFERENT K VALUES")
print("=" * 50)

print(results_df.round(4))

# Step 7: Analyze churn by contract type

print("\n" + "=" * 50)
print("CHURN ANALYSIS BY CONTRACT TYPE")
print("=" * 50)

contract_analysis = df.groupby("Contract")["Churn"].agg(
    ["count", "sum", "mean"]
)

contract_analysis.columns = [
    "Total Customers",
    "Churned Customers",
    "Churn Rate"
]

contract_analysis["Churn Rate"] = (
    contract_analysis["Churn Rate"] * 100
)

print(contract_analysis.round(2))

# Step 7: Additional churn analysis

for feature in ["InternetService", "PaymentMethod"]:

    print("\n" + "=" * 50)
    print(f"CHURN ANALYSIS BY {feature}")
    print("=" * 50)

    analysis = df.groupby(feature)["Churn"].agg(
        ["count", "sum", "mean"]
    )

    analysis.columns = [
        "Total Customers",
        "Churned Customers",
        "Churn Rate"
    ]

    analysis["Churn Rate"] = analysis["Churn Rate"] * 100

    print(analysis.round(2))

    # Step 7: Visualize key churn findings

import matplotlib.pyplot as plt

features = [
    "Contract",
    "InternetService",
    "PaymentMethod"
]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, feature in zip(axes, features):

    churn_rates = (
        df.groupby(feature)["Churn"].mean() * 100
    )

    churn_rates.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(f"Churn Rate by {feature}")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 100)

plt.tight_layout()

# Save chart for submission
plt.savefig("churn_analysis.png", dpi=300)

print("\nChurn analysis chart saved successfully!")

# Step 8: Visualize KNN performance

plt.figure(figsize=(10, 6))

plt.plot(
    results_df["K"],
    results_df["Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    results_df["K"],
    results_df["Precision"],
    marker="s",
    label="Precision"
)

plt.plot(
    results_df["K"],
    results_df["Recall"],
    marker="^",
    label="Recall"
)

plt.title("KNN Performance Comparison")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Score")
plt.xticks(k_values)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("churn_knn_comparison.png", dpi=300)

print("\nKNN comparison chart saved successfully!")


