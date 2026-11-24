#Part 2: Unguided Exercise - 
# Customer Churn Prediction

#Step 1: Load and Explore the Data

"""
Telco Customer Churn Prediction with KNN
Author: [Your Name]
Description: Predict whether a telecommunications customer will churn using KNN
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ==========================================
# STEP 1: LOAD AND EXPLORE THE DATA
# ==========================================
CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(CSV_PATH)

print("Dataset shape (rows, columns):", df.shape)
print("\nColumn names and data types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nExplicit NaN count per column:")
print(df.isnull().sum()[df.isnull().sum() >= 0])

# Check blank/whitespace strings in object columns (common in TotalCharges)
for col in df.select_dtypes(include="object").columns:
    blank_count = (df[col].astype(str).str.strip() == "").sum()
    if blank_count > 0:
        print(f"Column '{col}' has {blank_count} blank/whitespace entries")

print("\nTarget ('Churn') distribution:")
print(df["Churn"].value_counts())
print("\nTarget ('Churn') proportions:")
print(df["Churn"].value_counts(normalize=True).round(4))

# Summary statistics for numeric columns
print("\nNumeric summary statistics:")
print(df.describe())

# Visualize key numeric features and churn counts
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

df["Churn"].value_counts().plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
axes[0].set_title("Churn Count (No vs. Yes)")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Customers")
axes[0].tick_params(axis="x", rotation=0)

for label, grp in df.groupby("Churn"):
    axes[1].hist(grp["tenure"], bins=20, alpha=0.6, label=f"Churn={label}")
axes[1].set_title("Tenure Distribution by Churn")
axes[1].set_xlabel("Tenure (months)")
axes[1].set_ylabel("Customers")
axes[1].legend()

for label, grp in df.groupby("Churn"):
    axes[2].hist(grp["MonthlyCharges"], bins=20, alpha=0.6, label=f"Churn={label}")
axes[2].set_title("MonthlyCharges Distribution by Churn")
axes[2].set_xlabel("Monthly Charges ($)")
axes[2].set_ylabel("Customers")
axes[2].legend()

plt.tight_layout()
plt.savefig("churn_eda_overview.png", dpi=150, bbox_inches="tight")
print("\nSaved visualization to 'churn_eda_overview.png'")

# ==========================================
# STEP 2: DATA PREPROCESSING
# ==========================================

# ==========================================
# STEP 2: DATA PREPROCESSING
# ==========================================
print("\n" + "=" * 50)
print("STEP 2: DATA PREPROCESSING")
print("=" * 50)

df_clean = df.copy()

# 1. Convert TotalCharges to numeric (blank strings become NaN)
df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
nan_total_charges = df_clean["TotalCharges"].isna().sum()
print(f"TotalCharges NaNs after numeric conversion: {nan_total_charges}")

# Fill any remaining missing numeric values with median (or 0 for new zero-tenure rows)
if nan_total_charges > 0:
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(
        df_clean["TotalCharges"].median()
    )

# Fill any other missing values if present
for col in df_clean.select_dtypes(include=[np.number]).columns:
    if df_clean[col].isna().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in df_clean.select_dtypes(include="object").columns:
    if df_clean[col].isna().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# 2. Drop customerID (identifier, not a predictive feature)
if "customerID" in df_clean.columns:
    df_clean = df_clean.drop(columns=["customerID"])

# 3. Convert target 'Churn' to binary (1 = Yes, 0 = No)
y = df_clean["Churn"].map({"Yes": 1, "No": 0})
if y.isna().sum() > 0:
    raise ValueError("Unexpected values found in 'Churn' column.")

# 4. Separate raw features and encode categorical columns to numeric
X_raw = df_clean.drop(columns=["Churn"])
X = pd.get_dummies(X_raw, drop_first=True, dtype=int)

print(f"Processed features shape (X): {X.shape}")
print(f"Processed target shape (y):   {y.shape}")
print(f"Any remaining NaNs in X: {X.isna().sum().sum()} | in y: {y.isna().sum()}")
print(f"Churn target counts (1=Yes, 0=No):\n{y.value_counts()}")


# ==========================================
# STEP 3: SPLIT THE DATA (80% TRAIN / 20% TEST)
# ==========================================
print("\n" + "=" * 50)
print("STEP 3: TRAIN / TEST SPLIT")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training rows: {X_train.shape[0]} ({X_train.shape[0] / len(X):.1%})")
print(f"Testing rows:  {X_test.shape[0]} ({X_test.shape[0] / len(X):.1%})")
print("\nTraining Churn distribution (1=Yes, 0=No):")
print(y_train.value_counts(normalize=True).round(4))
print("\nTesting Churn distribution (1=Yes, 0=No):")
print(y_test.value_counts(normalize=True).round(4))

#Part 2 — Step 4: Train a KNN Model (n_neighbors=5)

# ==========================================
# STEP 4: TRAIN A KNN MODEL (K=5)
# ==========================================
print("\n" + "=" * 50)
print("STEP 4: TRAINING KNN MODEL (n_neighbors=5)")
print("=" * 50)

churn_knn = KNeighborsClassifier(n_neighbors=5)
churn_knn.fit(X_train, y_train)

y_train_pred_churn = churn_knn.predict(X_train)
y_test_pred_churn = churn_knn.predict(X_test)

print("Churn KNN classifier trained successfully!")
print(f"Neighbors (K): {churn_knn.n_neighbors}")
print(f"Training predictions shape: {y_train_pred_churn.shape}")
print(f"Testing predictions shape:  {y_test_pred_churn.shape}")
print("Predicted churn counts on test set (1=Yes, 0=No):")
print(pd.Series(y_test_pred_churn).value_counts())


# ==========================================
# STEP 5: MAKE PREDICTIONS AND EVALUATE (K=5)
# ==========================================
print("\n" + "=" * 50)
print("STEP 5: MODEL EVALUATION (CHURN TEST SET, K=5)")
print("=" * 50)

churn_train_acc = accuracy_score(y_train, y_train_pred_churn)
churn_test_acc = accuracy_score(y_test, y_test_pred_churn)

# pos_label=1 evaluates churners ("Yes"), which is the positive target event
churn_precision = precision_score(y_test, y_test_pred_churn, pos_label=1, zero_division=0)
churn_recall = recall_score(y_test, y_test_pred_churn, pos_label=1, zero_division=0)

print(f"Training Accuracy: {churn_train_acc:.4f} ({churn_train_acc * 100:.2f}%)")
print(f"Test Accuracy:     {churn_test_acc:.4f} ({churn_test_acc * 100:.2f}%)")
print(f"Churn (1) Precision: {churn_precision:.4f} ({churn_precision * 100:.2f}%)")
print(f"Churn (1) Recall:    {churn_recall:.4f} ({churn_recall * 100:.2f}%)")

# Confusion Matrix with labels=[0, 1] -> rows=Actual, cols=Predicted
churn_cm = confusion_matrix(y_test, y_test_pred_churn, labels=[0, 1])
print("\nConfusion Matrix [rows=Actual, cols=Predicted | 0=No Churn, 1=Churn]:")
print(pd.DataFrame(
    churn_cm,
    index=["Actual No Churn (0)", "Actual Churn (1)"],
    columns=["Pred No Churn (0)", "Pred Churn (1)"],
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_test_pred_churn,
    target_names=["No Churn (0)", "Churn (1)"],
    zero_division=0,
))

# Part 2 — Step 6: Experiment and
#Improve (K in [1, 3, 5, 7, 9, 11, 15])

# ==========================================
# STEP 6: EXPERIMENT WITH DIFFERENT K VALUES
# ==========================================
print("\n" + "=" * 50)
print("STEP 6: COMPARING K VALUES FOR CHURN PREDICTION")
print("=" * 50)

churn_k_values = [1, 3, 5, 7, 9, 11, 15]
churn_results = []

for k in churn_k_values:
    model_k = KNeighborsClassifier(n_neighbors=k)
    model_k.fit(X_train, y_train)

    tr_acc_k = accuracy_score(y_train, model_k.predict(X_train))
    y_pred_k = model_k.predict(X_test)
    te_acc_k = accuracy_score(y_test, y_pred_k)
    ch_prec_k = precision_score(y_test, y_pred_k, pos_label=1, zero_division=0)
    ch_rec_k = recall_score(y_test, y_pred_k, pos_label=1, zero_division=0)
    fn_k = confusion_matrix(y_test, y_pred_k, labels=[0, 1])[1, 0]

    churn_results.append({
        "K": k,
        "Train_Acc": round(tr_acc_k, 4),
        "Test_Acc": round(te_acc_k, 4),
        "Churn_Prec": round(ch_prec_k, 4),
        "Churn_Rec": round(ch_rec_k, 4),
        "False_Neg": fn_k,
    })

churn_results_df = pd.DataFrame(churn_results)
print(churn_results_df.to_string(index=False))

# Choose best K by test accuracy (tie-breaking with higher churn recall)
best_churn_row = churn_results_df.sort_values(
    by=["Test_Acc", "Churn_Rec"], ascending=[False, False]
).iloc[0]
best_churn_k = int(best_churn_row["K"])

print(
    f"\nSelected K = {best_churn_k} -> "
    f"Test Accuracy = {best_churn_row['Test_Acc']:.4f}, "
    f"Churn Precision = {best_churn_row['Churn_Prec']:.4f}, "
    f"Churn Recall = {best_churn_row['Churn_Rec']:.4f}, "
    f"False Negatives = {int(best_churn_row['False_Neg'])}"
)

# Plot accuracy, churn precision, and churn recall across K
plt.figure(figsize=(10, 6))
plt.plot(churn_results_df["K"], churn_results_df["Test_Acc"], marker="o", label="Test Accuracy")
plt.plot(churn_results_df["K"], churn_results_df["Churn_Prec"], marker="s", label="Churn (1) Precision")
plt.plot(churn_results_df["K"], churn_results_df["Churn_Rec"], marker="^", label="Churn (1) Recall")
plt.xticks(churn_k_values)
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Score")
plt.title("Telco Churn KNN Performance vs. K Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("churn_k_comparison.png", dpi=150, bbox_inches="tight")
print("Saved visualization to 'churn_k_comparison.png'")

#Part 2 — Step 7: Feature Signals & 
# Final Script Summary 

# ==========================================
# STEP 7: ANALYSIS AND RECOMMENDATIONS
# ==========================================
print("\n" + "=" * 50)
print("STEP 7: FEATURE CORRELATIONS & FINAL SUMMARY")
print("=" * 50)

# 1. Inspect features most strongly correlated with Churn (1 = Yes, 0 = No)
correlations = X.apply(lambda col: col.corr(y)).sort_values()

print("\nTop 5 features negatively associated with Churn (lower churn risk):")
print(correlations.head(5).round(4))

print("\nTop 5 features positively associated with Churn (higher churn risk):")
print(correlations.tail(5).iloc[::-1].round(4))

# Plot top positive and negative correlations with Churn
top_corr = pd.concat([correlations.head(5), correlations.tail(5)])
plt.figure(figsize=(9, 5))
colors = ["#2b8cbe" if val < 0 else "#e34a33" for val in top_corr.values]
top_corr.plot(kind="barh", color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Pearson Correlation with Churn (1 = Yes)")
plt.title("Top Features Associated with Customer Churn")
plt.tight_layout()
plt.savefig("churn_feature_correlations.png", dpi=150, bbox_inches="tight")
print("\nSaved visualization to 'churn_feature_correlations.png'")

# 2. Concise console summary of findings
print("\n--- Final Summary ---")
print(f"Baseline K=5 Test Accuracy: {churn_test_acc:.4f} | Churn Recall: {churn_recall:.4f}")
print(
    f"Best K={best_churn_k} Test Accuracy: {best_churn_row['Test_Acc']:.4f} | "
    f"Churn Precision: {best_churn_row['Churn_Prec']:.4f} | "
    f"Churn Recall: {best_churn_row['Churn_Rec']:.4f}"
)
print("Key Churn Drivers: Short tenure, month-to-month contracts, fiber optic service, electronic check payment.")
print("Key Retention Factors: Longer tenure, two-year contracts, online security / tech support add-ons.")

