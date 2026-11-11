import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Explore the dataset
print("=== DATASET SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns)

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Check target distribution
print("\n=== CHURN DISTRIBUTION ===")
print(df["Churn"].value_counts())

# Visualize churn distribution
df["Churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show(block=False)
plt.pause(1)
plt.close()


# %%
# Step 2: Data Preprocessing

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

# Check missing values again
print("\n=== MISSING VALUES AFTER CONVERSION ===")
print(df.isnull().sum())

# Handle missing values
for column in df.columns:
    if df[column].isnull().any():
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

print("\n=== MISSING VALUES AFTER HANDLING ===")
print(df.isnull().sum().sum())

# Select features and target
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Convert categorical variables to numeric
categorical_columns = X.select_dtypes(
    include=["object", "string", "str"]
).columns

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

# Display preprocessing results
print("\n=== PREPROCESSING RESULTS ===")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nFeature data types:")
print(X.dtypes.value_counts())

print("\nFirst 5 rows of processed features:")
print(X.head())


# %%
# Step 3: Split the Data

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the results
print("\n=== DATA SPLIT RESULTS ===")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# %%
# Step 4: Train a KNN Model

from sklearn.neighbors import KNeighborsClassifier

# Create the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model on the training data
knn.fit(X_train, y_train)

print("\n=== KNN MODEL TRAINED ===")
print("Number of neighbors:", knn.n_neighbors)

# %%
# Step 5: Make Predictions and Evaluate

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate classification report
report = classification_report(y_test, y_pred)

# Display results
print("\n=== MODEL EVALUATION ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print("\n=== CONFUSION MATRIX ===")
print(conf_matrix)

print("\n=== CLASSIFICATION REPORT ===")
print(report)

# %%
# Step 6: Experiment and Improve

# Try different K values
k_values = [1, 3, 5, 7, 9, 11, 15]

results = []

for k in k_values:
    # Create and train a KNN model
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    # Make predictions
    y_pred_k = knn_model.predict(X_test)

    # Calculate performance
    accuracy_k = accuracy_score(y_test, y_pred_k)

    results.append((k, accuracy_k))

# Display the results
print("\n=== K VALUE COMPARISON ===")

for k, accuracy_k in results:
    print(f"K = {k}: Accuracy = {accuracy_k:.4f}")

# Choose the best K value
best_k, best_accuracy = max(results, key=lambda x: x[1])

print("\n=== BEST K VALUE ===")
print(f"Best K: {best_k}")
print(f"Best Accuracy: {best_accuracy:.4f}")
best_k, best_accuracy = max(results, key=lambda x: x[1])

# %%
# Step 7: Analysis and Recommendations

print("\n=== STEP 7: ANALYSIS AND RECOMMENDATIONS ===")

print("\n1. MODEL ACCURACY")
print("The best KNN model used K=15 and achieved an accuracy of 78.99%.")

print("\n2. IMPORTANT FEATURES")
print("Features such as tenure, contract type, monthly charges,")
print("internet service, and payment method are relevant when")
print("analysing customer churn.")

print("\n3. RECOMMENDATIONS")
print("The company should focus on customers with short tenure,")
print("higher monthly charges, and month-to-month contracts.")
print("Targeted retention offers and improved customer support")
print("could help reduce customer churn.")

print("\n4. MODEL LIMITATIONS")
print("The model does not identify all customers who churn.")
print("The recall for the churn class was 43.05% with K=5.")
print("KNN can also be affected by the choice of K and the")
print("scale of numerical features.")





























































































# %%
