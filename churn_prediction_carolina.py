"""
LAB | sklearn Model Training + Evaluation
Part 2: Unguided Exercise - Customer Churn Prediction
Author: Carolina Uchoa

Dataset: Telco Customer Churn (Kaggle) - 7,043 customers.
Features: customer demographics, services subscribed, account information.
Target: whether the customer churned (Yes/No), mapped to 1 = churned, 0 = stayed.
The CSV must be in the same folder as this script.
"""


# STEP 1: Load and Explore the Data
##download the churn dataset
##load it into a pandas dataframe
##explore the dataset to understand its structure, features, and target variable
##check the target variable distribution and any missing values


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, classification_report)

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(df.shape)       # (7043, 21): 7,043 customers, 21 columns
print(df.dtypes)      # TotalCharges shows as text: it hides blank values
print(df.head())

# Blank TotalCharges values become NaN, so we can count them as missing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(df.isnull().sum().sum(), "missing values")   # 11, all in TotalCharges

# Target distribution: counts and percentages
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True).round(3) * 100)  # ~26.5% churn

print(df.describe())

# Visualize key features: churn rate by contract type and by internet service
churn_flag = (df['Churn'] == 'Yes').astype(int)
churn_flag.groupby(df['Contract']).mean().plot(kind='bar', title='Churn rate by contract')
plt.ylabel('Churn rate')
plt.show()
churn_flag.groupby(df['InternetService']).mean().plot(kind='bar', title='Churn rate by internet service')
plt.ylabel('Churn rate')
plt.show()
# Findings: month-to-month contracts (~43%) and fiber optic customers (~42%)
# churn the most; two-year contracts almost never churn (~3%).


# STEP 2: Data Preprocessing
#Handle missing values (if any)
#Convert categorical variables to numeric (use encoding techniques)
#Select relevant features for modeling
#Separate features (X) from target (y)
#Convert target to binary (0/1) if needed

df = df.dropna()
df = df.drop(columns=['customerID'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
 
# Turn all text columns into numbers (one-hot encoding)
df = pd.get_dummies(df, drop_first=True, dtype=int)
 
X = df.drop(columns='Churn')
y = df['Churn']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")


# STEP 3: Split the Data
#Split into training (80%) and testing (20%) sets
#Use stratify to maintain class distribution
#Set random_state=42 for reproducibility

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 50)
print("DATA SPLIT")
print("=" * 50)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Training features: {X_train.shape[1]}")
print(f"Test features: {X_test.shape[1]}")

# Verify churn distribution
print("\nTraining set churn distribution:")
print(y_train.value_counts())
print(f"  No churn (0): {(y_train == 0).sum()} "
      f"({(y_train == 0).mean() * 100:.1f}%)")
print(f"  Churn (1): {(y_train == 1).sum()} "
      f"({(y_train == 1).mean() * 100:.1f}%)")

print("\nTest set churn distribution:")
print(y_test.value_counts())
print(f"  No churn (0): {(y_test == 0).sum()} "
      f"({(y_test == 0).mean() * 100:.1f}%)")
print(f"  Churn (1): {(y_test == 1).sum()} "
      f"({(y_test == 1).mean() * 100:.1f}%)")


# STEP 4: Train the KNN Model
#Create KNN classifier with k=5
#Fit the model on the training data

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("KNN classifier trained successfully!")
print(f"Number of neighbors (k): {knn.n_neighbors}")
 
# Make predictions
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)
 
print(f"\nTraining predictions: {len(y_train_pred)}")
print(f"Test predictions: {len(y_test_pred)}")


#STEP 5: Make Predictions and Evaluate
#Create predictions on the test set
#Calculate accuracy, precision, recall, and confusion matrix
#Generate a classification report for detailed metrics
#Interpret the results

# Predictions were made in Step 4 (y_train_pred, y_test_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, zero_division=0)
recall = recall_score(y_test, y_test_pred, zero_division=0)
cm = confusion_matrix(y_test, y_test_pred)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=["No Churn", "Churn"],
    zero_division=0
))


# STEP 6: Experiment and Improve
#Try different values of k (e.g., 3, 5, 7, 9) and observe how the performance metrics change
#Compare their performances
#Chose the best k based on accuracy, precision, and recall
#Document your findings and any insights gained from the experiments

# Churn = 1 is the positive class, so these metrics measure churn detection
k_values = [3, 5, 7, 9]
experiment_results = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    experiment_results.append({
        'k': k,
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0)
    })

results_df = pd.DataFrame(experiment_results)
print("=== KNN Experiment Results ===")
print(results_df.round(3).to_string(index=False))

best_accuracy_k = int(results_df.loc[results_df['accuracy'].idxmax(), 'k'])
best_precision_k = int(results_df.loc[results_df['precision'].idxmax(), 'k'])
best_recall_k = int(results_df.loc[results_df['recall'].idxmax(), 'k'])

print("\n=== Findings to Document ===")
print(f"Best k for accuracy: {best_accuracy_k}")
print(f"Best k for precision: {best_precision_k}")
print(f"Best k for recall: {best_recall_k}")
print("Recommendation: prioritize the k with the highest recall when identifying customers at risk of churn.")

# Keep the accuracy-selected model available for the next step
best_k = best_accuracy_k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_test_pred = knn.predict(X_test)
print(f"\nFinal model uses k={best_k}.")


# STEP 7: ANALYSIS AND RECOMMENDATIONS
final_knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
final_pred = final_knn.predict(X_test)
baseline = 1 - y_test.mean()

# 1. Model accuracy (compared with the baseline)
print("=== 1. Model accuracy ===")
print(f"Final model (k=7) accuracy: {accuracy_score(y_test, final_pred):.3f}")
print(f"Baseline (always 'no churn'): {baseline:.3f}")
print(f"Recall (churners caught): {recall_score(y_test, final_pred):.3f}")

# 2. Most important features: correlation of each feature with Churn
print("\n=== 2. Features most related to churn ===")
corr = df.corr()['Churn'].drop('Churn').sort_values()
print("Linked to MORE churn:")
print(corr.tail(5).round(2))
print("\nLinked to LESS churn:")
print(corr.head(5).round(2))

corr.tail(5).plot(kind='barh', title='Top features linked to churn')
plt.show()

# 3. Recommendations and 4. Limitations
print("""
== 3. Recommendations ==
- Target retention at month-to-month, new and fiber optic customers.
- To offer incentives to move customers to 1- or 2-year contracts.
- Review the fiber optic offer (price and service quality).
- Encourage automatic payment methods to facilitate. Electronic-check customers churn more.
- Use the model as an early-warning list: a retention offer costs
  less than losing a customer, so some false alarms are acceptable.

== 4. Limitations ==
- Accuracy is only slightly above the baseline, and the model misses
  more than half of the churners (low recall).
- KNN does not explain WHY a customer churns.
- Results come from a single train/test split.
- Next steps: scale the features, try Logistic Regression or Random
  Forest, and use cross-validation.
""")
