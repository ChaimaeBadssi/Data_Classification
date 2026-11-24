# Data_Classification
Spam &amp; Diabetes Classification: A Comparative Study of Machine Learning Models. Comparison of KNN, Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting, and MLP on UCI Spambase and BRFSS 2015 datasets. Explores PCA, feature selection, log transformation, SMOTE, hyperparameter tuning, and model evaluation.

## What's in this repo

| File | Purpose |
|---|---|
| `general.py` | Shared module: one tuning function and one training function per model, plus evaluation and plotting helpers |
| `spambase.ipynb` | Spam detection: preprocessing, model tuning, comparison |
| `diabetes.ipynb` | Diabetes prediction on a heavily imbalanced dataset |
| `spambase.data`  | (UCI Spambase, 4601 emails, 57 features)|
|`diabetes_binary_health_indicators_BRFSS2015.csv`| (253,680 survey responses, 21 features) |


## The shared module (`general.py`) Workflow

We wrote the model code once and reused it in both notebooks. For each of the models there are two functions:

- `best_*` runs a small grid search, fits on the training set and scores on the **validation** set. It returns the best hyperparameters and the validation accuracy.
- `*_model` trains the final model with the chosen hyperparameters and returns `(model, test_accuracy)`.

| Model | Tuned hyperparameters |
|---|---|
| KNN | `k` |
| Logistic Regression | none (`max_iter=1000`) |
| SVM | `kernel` (linear, poly, rbf, sigmoid) and `C` |
| Decision Tree | `max_depth`, `min_samples_split` |
| Random Forest | `n_estimators`, `max_depth` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth` |
| MLP | `hidden_layer_sizes`, `alpha` |

Two helpers handle the comparison step:

- `evaluate_models(models, X_test, y_test)` returns a DataFrame with accuracy, precision, recall, F1, false positives and false negatives, sorted by F1.
- `plot_confusion_matrices(models, X_test, y_test)` draws one confusion matrix per model in a grid.

`test_general.py` builds a small synthetic dataset with `make_classification`, splits it into train/val/test, and checks that each function returns the right types, that the selected hyperparameters come from the lists we gave it, that the trained models carry the requested parameters, and that accuracies are in [0, 1]. The plotting test mocks `plt.show`.

---

## Project 1: Spambase

**Data.** 4601 emails described by 57 numeric features (word frequencies, character frequencies, capital-letter run lengths). About 39% are spam. There are no missing values but 391 duplicate rows, which we dropped.

**What we noticed**
- Features live on very different scales: `capital_run_length_total` is in the hundreds or thousands, word frequencies are around 0.1.
- Only three pairs of features are strongly correlated (`word_freq_857`, `word_freq_415`, `word_freq_direct`), but we still used PCA to compress the 57 features.
- Some features have extreme outliers (e.g. `word_freq_free`).

**Pipeline**
1. 70/15/15 stratified train/validation/test split (the spam/non-spam ratio is kept in each part).
2. `StandardScaler`, optionally preceded by `log(1 + x)` to tame the skew.
3. PCA keeping 95% of the variance.
4. Hyperparameters tuned on the validation set, final scores on the test set.

**Log transform.** We tested it on Logistic Regression first: accuracy went from 92.2% to 93.3%, so we used it for all the other models.

**Results on the test set**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9225 | 0.9177 | 0.8849 | 0.9010 |
| SVM (linear, C=0.1) | 0.9225 | 0.9212 | 0.8810 | 0.9006 |
| Gradient Boosting | 0.9193 | 0.9036 | 0.8929 | 0.8982 |
| MLP | 0.9146 | 0.8867 | 0.9008 | 0.8937 |
| Random Forest | 0.9161 | 0.9129 | 0.8730 | 0.8925 |
| Decision Tree | 0.9066 | 0.9038 | 0.8571 | 0.8798 |
| KNN (k=10) | 0.9035 | 0.9244 | 0.8254 | 0.8721 |

**Our choice: SVM.** Logistic Regression and SVM tie on accuracy, but for a spam filter the costly mistake is sending a real email to the spam folder, so precision matters more than recall. SVM and KNN have the best precision (KNN has the fewest false positives, 17, but misses the most spam with 44 false negatives). SVM is the better compromise: 19 false positives and 30 false negatives.

---

## Project 2: Diabetes (BRFSS 2015)

**Data.** 253,680 survey responses, 21 health and lifestyle indicators (blood pressure, BMI, general health, age, income...) and a binary target. After removing 24,206 duplicates we keep 229,474 rows, of which only **15.3% are diabetic**. This imbalance drives most of what follows.

**Feature selection instead of PCA.** Most features here are binary survey answers, and PCA would give us components nobody can interpret. We wanted to know which health factors matter, so we kept the original features and used mutual information (`SelectKBest`, k=10). The ten features kept:

`HighBP`, `HighChol`, `BMI`, `HeartDiseaseorAttack`, `PhysActivity`, `GenHlth`, `PhysHlth`, `DiffWalk`, `Age`, `Income`

The strongest linear correlations with the target are `GenHlth` (0.28), `HighBP` (0.25), `DiffWalk` (0.21) and `BMI` (0.21).

**Pipeline.** 70/15/15 stratified split, standard scaling, feature selection, then optionally **SMOTE** on the training set only (never on validation or test). SMOTE brings the training set from 24,742 diabetic / 152,834 healthy to 152,834 / 152,834.

**Results on the test set**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Gradient Boosting (SMOTE) | 0.832 | 0.411 | 0.473 | **0.440** |
| Logistic Regression (SMOTE) | 0.731 | 0.309 | 0.757 | 0.439 |
| Random Forest (SMOTE) | 0.719 | 0.300 | 0.764 | 0.431 |
| KNN | 0.751 | 0.302 | 0.599 | 0.401 |
| KNN (SMOTE) | 0.817 | 0.324 | 0.289 | 0.305 |
| Decision Tree (SMOTE) | 0.821 | 0.329 | 0.273 | 0.298 |
| Logistic Regression | 0.863 | 0.531 | 0.145 | 0.228 |
| Decision Tree | 0.864 | 0.553 | 0.141 | 0.225 |
| Gradient Boosting | 0.865 | 0.561 | 0.137 | 0.220 |
| MLP | 0.866 | 0.585 | 0.125 | 0.206 |
| Random Forest | 0.865 | 0.621 | 0.079 | 0.140 |

**What we take from it**
- Accuracy is misleading here. Predicting "not diabetic" for everyone already gives about 85%. The models without SMOTE reach 86.5% but catch only 8 to 15% of diabetic patients.
- SMOTE trades accuracy for recall. Logistic Regression and Random Forest go above 75% recall, at the cost of many false positives (around 9,000).
- The effect of SMOTE depends on the model, so it is not a universal fix.
- Gradient Boosting with SMOTE gives the best F1 (0.44) and the best balance. If the goal is screening, where missing a patient is worse than a false alarm, Logistic Regression or Random Forest with SMOTE are also defensible because of their recall.

