# Part 1: Breast Cancer Prediction

## Step 1: Load and Explore the Data

The breast cancer dataset was loaded using `load_breast_cancer()` from sklearn. It contains 569 samples and 30 numerical features describing characteristics of cell nuclei, such as radius, texture, perimeter, area, and smoothness.

The dataset contained 212 malignant tumors and 357 benign tumors. There were no missing values. The target labels were:

```text
0 = Malignant
1 = Benign
```

## Step 2: Data Exploration

Statistical summaries were calculated using `df.describe()`. The feature distributions showed that malignant tumors generally had larger values for features such as mean radius, mean perimeter, and mean area compared with benign tumors.

The visualizations showed noticeable differences between malignant and benign samples. This suggests that these features may be useful for predicting whether a tumor is malignant or benign.

## Step 3: Splitting the Data

The features were separated from the target variable. The dataset was split into 80% training data and 20% testing data using `train_test_split`.

The split resulted in:

- 455 training samples
- 114 testing samples
- 30 features

Stratification was used to preserve a similar proportion of malignant and benign cases in both datasets. A `random_state` of 42 was used to make the results reproducible.

## Step 4: Training the Initial KNN Model

A baseline K-Nearest Neighbors classifier was created with `n_neighbors=5`. The model was trained using the training data. This initial model was later compared with models using different K values.

KNN makes predictions by finding the closest training samples and assigning the most common class among those neighbors. In this case, the model used the measurements of nearby tumor samples to predict whether each tumor was malignant or benign.

## Step 5: Making Predictions and Evaluating the Model

The initial KNN model with K = 5 achieved an accuracy of 91.23% on the test set. The model achieved 86.36% precision and 90.48% recall for malignant tumors.

The confusion matrix was:

```text
[[38  4]
 [ 6 66]]
```

This means that:

- 38 malignant tumors were correctly identified.
- 4 malignant tumors were incorrectly classified as benign.
- 6 benign tumors were incorrectly classified as malignant.
- 66 benign tumors were correctly identified.

Recall is particularly important for this problem because it measures how many actual malignant tumors were detected. Missing a malignant tumor is more serious than incorrectly classifying a benign tumor as malignant.

Overall, the initial model performed well and exceeded the target accuracy of 90%. The K = 5 results were used as a baseline for comparison with the other K values.

## Step 6: Experimenting with Different K Values

Several K values were tested to determine how the number of neighbors affected model performance.

| K | Accuracy | Malignant Precision | Malignant Recall |
|---:|---:|---:|---:|
| 1 | 92.98% | 90.48% | 90.48% |
| 3 | 92.98% | 90.48% | 90.48% |
| 5 | 91.23% | 86.36% | 90.48% |
| 7 | 92.98% | 90.48% | 90.48% |
| 9 | 93.86% | 92.68% | 90.48% |
| 11 | 93.86% | 92.68% | 90.48% |

K = 9 and K = 11 produced the highest accuracy at 93.86%. Both values also produced the same malignant precision and recall. K = 9 was selected as the final value because it achieved the same performance as K = 11 while using fewer neighbors.

## Step 7: Final Analysis and Conclusion

The final KNN model using K = 9 achieved an accuracy of 93.86%. It achieved 92.68% precision and 90.48% recall for malignant tumors. Compared with the initial K = 5 model, accuracy improved from 91.23% to 93.86%, and malignant precision improved from 86.36% to 92.68%. These results show that the final model performed well and exceeded the target accuracy of 90%.

The visualizations suggested that features such as radius, perimeter, and area may be useful for distinguishing malignant tumors from benign tumors. The model could potentially support medical professionals by identifying cases that require further investigation, although it should not replace expert medical judgment.

One limitation is that the KNN model was trained without feature scaling. KNN relies on distances between observations, so features with larger numerical ranges may have a greater influence on the predictions. Another limitation is that the dataset contains a different number of malignant and benign samples. The model's performance may also change with a different train/test split or a different choice of K.

Overall, the KNN model provided a strong baseline for breast cancer classification. The final model exceeded the required accuracy and detected most malignant tumors, although further testing, feature scaling, and model comparison could potentially improve its performance.
