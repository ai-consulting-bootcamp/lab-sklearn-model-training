# Customer Churn Prediction with K-Nearest Neighbors (KNN)

**Lab 2.2: Scikit-Learn Model Training**


## 1. Executive Summary & Approach

The objective of this project was to build and evaluate a K-Nearest Neighbors (KNN) classification model to predict customer churn (`Yes` = 1, `No` = 0) using the Telco Customer Churn dataset (7,043 customer records across 21 raw columns).

The end-to-end machine learning workflow followed seven stages:

1. **Exploratory Data Analysis:** Inspected shapes, data types, summary statistics, missing/whitespace values, and class proportions (`73.46%` non-churners vs. `26.54%` churners).

2. **Data Preprocessing:** Converted `TotalCharges` from string to numeric (`pd.to_numeric(..., errors="coerce")`), imputed the 11 resulting `NaN` entries with the column median, removed the non-predictive `customerID` column, mapped `Churn` to binary (`1`/`0`), and one-hot encoded categorical variables (`pd.get_dummies(..., drop_first=True)`), producing a clean feature matrix of **7,043 rows and 30 numeric features**.

3. **Train/Test Split**: Divided the data into **80% training (5,634 samples)** and **20% testing (1,409 samples)** with `stratify=y` and `random_state=42`, preserving the exact `73.46% / 26.54%` class distribution in both sets.

4. **Model Training & Tuning**: Trained a baseline `KNeighborsClassifier(n_neighbors=5)` and compared `K` values across `[1, 3, 5, 7, 9, 11, 15]`.





## 2. Model Performance Metrics (Telco Customer Churn)

### Baseline Model (`K = 5`)

* **Training Accuracy:** 82.78%
* **Test Accuracy:** 76.51% (1,078 of 1,409 test customers classified correctly)
* **Churn (`1`) Precision:** 57.71% (161 true churners out of 279 predicted churners)
* **Churn (`1`) Recall:** 43.05% (161 true churners caught out of 374 actual churners)
* **Churn (`1`) F1-Score:** 0.49

### Confusion Matrix (`K = 5`, Test Set = 1,409 Customers)

 | Predicted No Churn (`0`) | Predicted Churn (`1`) 
 **Actual No Churn (`0`)** | 917 (True Negatives) | 118 (False Positives) 

 **Actual Churn (`1`)** | 213 (False Negatives) | 161 (True Positives) 

### Hyperparameter Tuning (`K = 1, 3, 5, 7, 9, 11, 15`)

* Small `K` values (`K=1`, `K=3`) fit local noise closely, yielding higher training accuracy but lower test stability.
* Larger `K` values smooth the decision boundary; because `73.46%` of customers belong to the `No Churn (0)` majority class, higher `K` values increase overall accuracy slightly while reducing churn (`1`) recall.

* **Selected Best `K`:** `K = [PASTE BEST K FROM STEP 6]` achieved **[PASTE TEST ACCURACY]% test accuracy**, **[PASTE CHURN PRECISION]% churn precision**, and **[PASTE CHURN RECALL]% churn recall**.



## 3. Key Findings About Customer Churn

Correlation and feature-distribution analysis revealed clear behavioral drivers separating churned and retained customers:
* **Highest Churn Risk Factors:** Short customer `tenure`, **month-to-month contracts**, **fiber optic internet service**, **electronic check payment methods**, and higher `MonthlyCharges`.
* **Strongest Retention Factors:** Longer `tenure`, **two-year and one-year contracts**, and value-added support services such as **Online Security** and **Tech Support**.

* **Class Imbalance Impact:** Because `73.46%` of customers do not churn, a naive model predicting `No Churn` for everyone achieves `73.46%` accuracy with `0%` churn recall. While `K=5` beats the naive baseline at `76.51%` accuracy, `213` of `374` actual churners were missed (`43.05%` recall).



## 4. Business Recommendations

1. **Early-Tenure Onboarding Campaigns:** Focus proactive retention outreach and check-ins on customers in their first **1 to 12 months**, where churn risk is concentrated.

2. **Contract Upgrade Incentives:** Offer targeted discounts or complimentary `OnlineSecurity` / `TechSupport` add-ons to incentivize month-to-month customers to switch to 1-year or 2-year contracts.

3. **Fiber Optic & Electronic Check Experience Audit:** Investigate pricing friction, service reliability, or billing pain points among fiber-optic and electronic-check users, and promote automated payment enrollment.


## 5. Challenges, Limitations & Future Improvements

* **Challenges Encountered:** `TotalCharges` loaded as text due to 11 whitespace strings for zero-tenure customers, requiring numeric coercion and median imputation before modeling. Additionally, Python environment package installation and blocking `plt.show()` calls required terminal troubleshooting.
* **Model Limitations:

  1. *Unscaled Feature Distances:* Raw `TotalCharges` (up to `$8,684.80`) spans a much wider numeric range than `tenure` (`0–72`) or binary one-hot indicators (`0`/`1`), dominating Euclidean distance calculations in KNN.

  2. *Majority-Class Bias:* Class imbalance (`73.46%` vs. `26.54%`) pulls majority votes toward `No Churn (0)`, suppressing recall on churners.
 