


Question: Do we need to encode binary variable?

Treat truly numeric features with a numeric pipeline (median imputation, robust scaling, and optional numeric feature selection).
Treat multi-category nominal features (5–7 unique categories) with a categorical pipeline (most frequent imputation, one-hot encoding, and optional feature selection by Chi square
  or mutual information).
Treat binary features (already 0/1) either as-is (pass through) or with a minimal pipeline (e.g., simple imputation if missing). Typically, you won’t scale them or run one-hot encoding on them, since they’re already 0/1.


Question: Why Stratisfied Kfold in GridSearchCV?

Cross-Validation Splits:
When you run GridSearchCV or cross_val_score, your data is split into multiple “folds” (subsets) for training and validation in each iteration. If you use StratifiedKFold, each fold is created so that the original ratio of classes (fraud vs. non-fraud) is preserved as closely as possible.

Undersampling Occurs After the Fold Is Selected:

In each fold, you take a portion of the data as the “training fold” and another portion as the “validation fold.”
Only the training portion of that fold is then passed through your pipeline, which includes the undersampling step (NearMiss in your example).
The validation fold remains untouched by the resampler because you want to test on real-world distribution.
Thus, before the resampler is applied, the training data that goes into the pipeline is selected in a way that maintains the original class distribution within that fold—this is exactly what StratifiedKFold does.


Question: Explaining pipeline ?
# **Step-by-Step Pipeline Explanation**

Below is a detailed explanation of each step in your fraud detection pipeline. You can include this Markdown in your project documentation or report.

---

## **1. Data Splitting**

1. **Purpose**  
   - Separate your dataset into training and test sets to train the model on one subset and evaluate it on unseen data.
1. **Process**  
   - Typically done via `train_test_split(X, y, test_size=..., stratify=y, random_state=...)`.
   - This results in `(X_train, y_train)` for training and `(X_test, y_test)` for final evaluation.

> **Key Benefit**: Prevents *data leakage* by ensuring no information from the test set influences model training or hyperparameter tuning.

---

## **2. ColumnTransformer (Preprocessing)**

1. **Objective**  
   - Different feature types (numeric vs. categorical) require distinct transformations.

1. **Numeric Pipeline**
   - **Imputation**: `SimpleImputer(strategy='median')` handles missing numeric values by replacing them with the median.
   - **Scaling**: `StandardScaler` (or similar) brings numeric features to a comparable scale.
   - **Feature Selection**: `SelectKBest(score_func=f_classif)` picks the most informative numeric features based on statistical tests.

1. **Categorical Pipeline**
   - **Imputation**: `SimpleImputer(strategy='most_frequent')` replaces missing categories with the most common value.
   - **Encoding**: `OneHotEncoder(handle_unknown='ignore')` transforms categorical features into binary indicator columns.
   - **Feature Selection**: `SelectKBest(score_func=chi2)` selects the best categorical features via a chi-square test.

1. **Combining Pipelines**  
   - A `ColumnTransformer` applies the numeric pipeline to numeric columns and the categorical pipeline to categorical columns, merging outputs into a final numeric matrix.

---

## **3. Resampling (e.g., NearMiss, SMOTE, etc.)**

1. **Why Resample?**  
   - Fraud detection typically has a **highly imbalanced** dataset (few frauds vs. many non-frauds).

1. **NearMiss (Undersampling)**
   - Systematically removes majority-class samples to match the minority-class level, ensuring the model sees a more balanced dataset.
   - Helps the model focus on minority examples (fraud), possibly increasing recall—but can lead to higher false positives or loss of important majority information if too aggressive.

1. **Pipeline Placement**  
   - Occurs **after** preprocessing but **before** classifier training. In each fold of cross-validation, only the training portion is resampled, never the validation/test set.

---

## **4. Classifier**

1. **Choice of Algorithm**  
   - Started with `LogisticRegression` (or advanced models like `RandomForest`, `XGBClassifier`), which learns how to classify transactions as fraud or not.
1. **Why in a Pipeline?**
   - `(preprocessing) -> (resampler) -> (classifier)` ensures every training fold in cross-validation goes through the same transformations and class balancing steps.
   - Consistency across folds prevents data leakage and ensures the final test evaluation is fair.

---

## **5. Custom F2 Scorer**

1. **Definition**  
   - `make_scorer(fbeta_score, beta=2)` emphasizes recall more than precision while still including precision in the F2 calculation.
1. **Motivation**  
   - Missing a fraud (false negative) can be costlier than flagging a legitimate transaction (false positive), so boosting recall is crucial.
1. **Usage**  
   - Passed as `scoring=custom_scorer` in `GridSearchCV`, guiding hyperparameter tuning to maximize F2.

---

## **6. GridSearchCV with StratifiedKFold**

1. **Cross-Validation**  
   - **StratifiedKFold** ensures each fold in the training set maintains the original class ratio, yielding more stable performance estimates.
1. **Hyperparameter Search**  
   - `GridSearchCV` tries multiple settings (e.g., feature selection `k`, classifier parameters).  
   - In each fold:
     1. **Train** pipeline on the training fold (preprocessing + undersampling + classification).
     2. **Validate** on the remaining fold (untouched, real distribution).
   - Picks the combination that yields the best F2 average across folds.
1. **Refit**  
   - After finding the best hyperparameters, the pipeline refits on the entire `X_train, y_train` data.

---

## **7. Final Evaluation on Test Data**

1. **Consistency**  
   - The pipeline applies the same numeric/categorical transformations to `X_test` (but does not resample).
1. **Real-World Distribution**  
   - The test set remains imbalanced, reflecting real fraud rates—no undersampling here.
1. **Metrics**  
   - Compare predicted labels with `y_test`.
   - Use `classification_report` or confusion matrix for recall, precision, F2, false positive rate (FPR), etc.
1. **Optional Threshold Adjustment**  
   - If FPR or recall is unsatisfactory, adjust the decision threshold (e.g., based on predicted probabilities) to meet business needs.

---

## **8. Overall Workflow Summary**

1. **Split** data into train/test.
2. **Preprocess** features using:
   - Numeric pipeline (impute, scale, select).
   - Categorical pipeline (impute, encode, select).
3. **Resample** (e.g., `NearMiss`) on the training fold to balance classes.
4. **Train** the classifier (LogisticRegression or advanced models).
5. **Tune** hyperparameters with `GridSearchCV` and a **custom F2 scorer**:
   - Uses `StratifiedKFold` to keep class ratios consistent per fold.
   - Chooses the best combination of feature selection `k`, resampler settings (if any), and classifier parameters.
6. **Evaluate** the final best pipeline on the untouched test set, measuring recall, F2, FPR, etc.
7. (Optional) **Adjust** thresholds if you need a lower FPR or higher recall, depending on the business trade-offs.

You now have a **comprehensive** end-to-end solution for fraud detection, leveraging robust preprocessing, undersampling to handle imbalance, a carefully tuned classifier, and an F2 objective that prioritizes capturing fraud without completely sacrificing precision.






# Classification Report (Threshold = 0.5)



## 1. Explanation in Simple Terms

- **Accuracy (13%)**  
  Overall, the model is correct only 13% of the time. Because most data points (nearly 198k) are Non-Fraud, predicting “Fraud” too often hurts accuracy.

- **Precision**  
  - **Non-Fraud (0.99)**: Of the samples predicted *Non-Fraud*, 99% really are *Non-Fraud*.  
  - **Fraud (0.01)**: Of the samples predicted *Fraud*, only 1% are truly *Fraud* (i.e., 99% are false alarms).

- **Recall**  
  - **Non-Fraud (0.12)**: Of all actual Non-Fraud cases, the model only correctly labels 12% of them as Non-Fraud. It mistakenly flags most as Fraud.  
  - **Fraud (0.88)**: Of all actual Fraud cases, 88% are caught by the model (it has high recall for Fraud).

- **F1-Score**  
  - **Non-Fraud (0.22)**: Because recall is low (0.12), the F1 drops.  
  - **Fraud (0.02)**: Despite a high recall (0.88), the model’s Fraud precision is very low (0.01), so the combined F1 is just 0.02.

## 2. Confusion Matrix Concept

|                     | Predicted Non-Fraud | Predicted Fraud | Total (Actual)  |
|---------------------|---------------------|-----------------|-----------------|
| **Actual Non-Fraud** | TN (23,700) | FP (174,094.)  | 197,794         |
| **Actual Fraud**     | FN (265)| TP (1,941)   | 2,206           |
| **Total (Predicted)** | 	23,965                  |      176,035           | 200,000         |

From such a matrix, we calculate:

### Accuracy

\[
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{All Samples}}
\]

### Precision (for a class)

\[
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
\]

### Recall (for a class)

\[
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
\]

### F1 Score

\[
F1 = 2 \times \frac{(\text{Precision} \times \text{Recall})}{(\text{Precision} + \text{Recall})}
\]

## 3. Summary

- The model **rarely misses a Fraud** (88% recall) but **labels nearly everything as Fraud**, producing many false positives.  
- You get **very low Fraud precision** (1%), which drives overall accuracy down to 13%.  
- In fraud detection, such a model would catch most fraud but at the cost of investigating a huge number of legitimate applications.

--------------------------------------------------------------------------






# Summary: Decoupling ML Models from Business Decision Thresholds via Calibration

## **1. What is Decoupling?**
Decoupling refers to **separating the machine learning (ML) model’s output scores from the business decision thresholds**. This ensures that **business rules remain stable** even as the ML model undergoes updates or retraining.

## **2. Why is Decoupling Important?**
- **Consistency**: Maintains stable decision-making criteria over time.
- **Compliance**: Simplifies adherence to regulatory standards by keeping thresholds fixed.
- **Efficiency**: Reduces the need for frequent threshold re-tuning, saving time and minimizing errors.

## **3. The Role of Calibration**
Calibration **aligns the ML model’s raw scores with true probability estimates**, ensuring that a score (e.g., 0.7) accurately reflects the likelihood of an event (e.g., 70% chance of fraud).

### **How Calibration Works Mathematically**
1. **Calibration Function (\(\phi\))**:
   - Maps raw scores \(s \in [0, 1]\) to calibrated probabilities \(\hat{p} \in [0, 1]\).
   - Example (Platt Scaling):
     \[
     \phi(s) = \frac{1}{1 + \exp(A \cdot s + B)}
     \]
     where \(A\) and \(B\) are learned parameters.

2. **Applying Calibration**:
   - **Initial Calibration**:
     \[
     \hat{p} = \phi(s)
     \]
   - **Post-Retraining**:
     - When retraining the model, **keep \(\phi\) fixed**.
     - New raw scores \(s'\) are transformed as:
       \[
       \hat{p}' = \phi(s')
       \]
     - This ensures that **\(\hat{p}'\)** remains consistent with the original probability interpretation.

## **4. Ensuring Stable Probability Outputs**
- **Fixed \(\phi\)**: By not updating the calibration function during retraining, the mapping from raw scores to probabilities remains unchanged.
- **Small Distribution Shifts**: If retraining introduces minor changes in score distributions, the fixed \(\phi\) continues to provide reliable probability estimates.
- **Threshold Stability**: A business threshold set on \(\hat{p}\) (e.g., \(\hat{p} \geq 0.3\)) remains meaningful and consistent across model updates.

## **5. Practical Implications for Project**
- **Initial Setup**:
  1. **Train** the ML model on the training set.
  2. **Calibrate** the model using the validation set to learn \(\phi\).
  3. **Set** the business threshold based on calibrated probabilities.

- **During Retraining**:
  1. **Retrain** the ML model with new data.
  2. **Apply** the fixed calibration function \(\phi\) to new raw scores.
  3. **Maintain** the original business threshold without re-tuning.

- **Benefits**:
  - **Consistent Decision-Making**: Business rules don’t fluctuate with model changes.
  - **Regulatory Ease**: Fixed thresholds simplify compliance and auditing processes.
  - **Operational Efficiency**: Reduces the overhead of frequent threshold adjustments.

## **Conclusion**
Calibration is a crucial step for decoupling ML models from business decision thresholds. By ensuring that probability estimates remain consistent despite model retraining and data shifts, calibration fosters **stable, reliable, and compliant decision-making** in dynamic environments.

-------------------------------------------------------------


# Why We Use Bootstrapping in Training

Bootstrapping is a fundamental technique employed in our experiment for several key reasons:

1. **Assessing Model Stability**
   - **Purpose**: Evaluates how consistent the model's performance is across different subsets of the training data.
   - **Benefit**: Identifies variability in predictions, ensuring the model doesn't overfit to a specific training sample.

2. **Estimating Performance Variability**
   - **Purpose**: Provides a distribution of performance metrics (e.g., Precision, Recall) by training multiple models on varied samples.
   - **Benefit**: Offers robust confidence intervals, enhancing the reliability of our evaluation.

3. **Enhancing Generalization**
   - **Purpose**: Simulates different training scenarios by sampling with replacement, exposing the model to diverse data patterns.
   - **Benefit**: Improves the model's ability to generalize to unseen data, reducing the risk of poor performance on new distributions.

4. **Facilitating Hyperparameter Tuning**
   - **Purpose**: Allows hyperparameters to be optimized on a representative subset, ensuring they perform well across multiple bootstrapped samples.
   - **Benefit**: Prevents overfitting during tuning, leading to more effective and transferable model configurations.

5. **Supporting Decoupling Strategies**
   - **Purpose**: Works in tandem with calibration techniques to maintain consistent decision thresholds despite variations in model training.
   - **Benefit**: Ensures that business logic remains stable and reliable, even as the underlying model evolves with new data.

## **Conclusion**

By integrating bootstrapping into our training protocol, we ensure that our fraud detection models are **robust**, **reliable**, and **consistent**. This approach not only aligns with the methodologies proposed in [Jesus et al., 2022](#) but also reinforces our commitment to maintaining high-performance standards in dynamic, real-world scenarios.


project/
├─ src/
│  ├─ data_extraction.py
│  ├─ data_validation.py
│  ├─ prep.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ validate_model.py
├─ pipelines/
│  └─ train_pipeline.py        # orchestrates the steps above
├─ serving/
│  ├─ app.py                   # FastAPI + load_model_from_registry()
│  └─ Dockerfile
├─ configs/
│  ├─ dev.yaml
│  ├─ staging.yaml
│  └─ prod.yaml
└─ deploy/
   ├─ airflow_dag.py / prefect_flow.py (if used)
   └─ ci_cd.yaml (promote model → redeploy service)
