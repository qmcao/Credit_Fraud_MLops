# Credit Card Application Project - Minh Cao

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Data & Preprocessing](#data--preprocessing)
4. [Model Development](#model-development)
6. [Deployment](#deployment)
7. [Usage](#usage)
9. [Future Work](#future-work)
10. [Contributing](#contributing)
11. [License](#license)
12. [Contact](#contact)




## Project Overview
In today’s rapidly evolving financial landscape, **credit card fraud** poses a persistent threat to both financial institutions and consumers, resulting in substantial monetary losses and undermining public trust. Accurately detecting fraudulent credit card applications is crucial, but **traditional fraud detection models** often fail to maintain their performance over time when confronted with new, shifting fraud patterns. Moreover, these models typically require **frequent threshold tuning**—an approach that is both **time-consuming** and **risky**, as any threshold change can introduce **inconsistency** in business operations and **complicate regulatory compliance.**

To overcome these challenges, this project explores **decoupling techniques** as proposed in recent research **[Luzio et al., 2024]**, which separate the **business decision threshold** from the **model’s evolving scores**. By **calibrating** model outputs into probability-like estimates and **fixing** a single business threshold, institutions can **adapt** to fresh data with minimal overhead while maintaining consistent, transparent decision criteria. Through this method, the system remains robust against fraud tactics that are continually morphing, sustaining a higher level of accuracy and reliability without the burden of incessant threshold re-tuning.

The primary objectives of this project are:

#### 1. Fraud Detection Model Development:
Utilize comprehensive Exploratory Data Analysis (EDA), data cleaning, feature engineering, and advanced machine learning techniques to develop a highly accurate model for identifying fraudulent credit card applications.

#### 2. Hyperparameter Optimization and Calibration:
Employ advanced optimization techniques, such as **Optuna**, to identify optimal hyperparameters for the LightGBM classifier. Subsequently, apply **calibration** methods to align predicted probabilities with true outcome frequencies, ensuring reliable threshold-based decision-making.

#### 3. Machine Learning Pipeline Development
Develop an **automated machine learning pipeline** comprising the following components:

- **Data_ingestion.py**: Automates the ingestion of data from various sources, ensuring seamless data flow into the system.

- **Data_transformation.py**: Automates data preprocessing tasks, including data cleaning, imputation of missing values, feature transformation, scaling, and encoding, to prepare data for model training.

- **model_training.py**: Automates the model training process, incorporating steps for hyperparameter tuning, model calibration, and threshold determination, thereby ensuring consistency and efficiency in model updates.

#### 4. Decoupling: 
Implement a **decoupling strategy**, as advocated by recent research [Luzio et al., 2024], to **separate** the evolving ML model from the **fixed business threshold**. Instead of continually re-tuning the threshold each time the model is retrained, this approach **calibrates** the model’s scores into reliable probabilities, ensuring that the **original cutoff** consistently meets target metrics (e.g., specific recall or FPR) over time. By keeping the threshold intact while updating only the underlying model, the system can accommodate new fraud patterns **without** frequent threshold adjustments or disruptions in established performance criteria.

#### 5. Performance Evaluation and Monitoring:
Assess model performance using key metrics— **recall, and false positive rates**—on both validation and test datasets.

Fureture goal: Implement monitoring tools to detect performance drift and trigger retraining processes when necessary.

## Project Structure
```text
Credit_Fraud_endToEnd/
└── artifacts/
    ├── data.csv
    ├── model.pkl
    ├── preprocessor.pkl
    ├── test.csv
    └── train.csv

└── notebook/
    ├── data/
    │   ├── fraud.csv
    ├── 1. EDA.ipynb
    ├── 2. MODEL_TRAINING.ipynb
    └── 3. MODEL DECOUPLING.ipynb

└── src/
    ├── components/
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   ├── model_trainer.py
    ├── pipeline/
    │   ├── predict_pipeline.py
    │   ├── train_pipeline.py
    ├── exception.py
    ├── logger.py
    └── utils.py

```


## Data & Preprocessing
This section outlines the comprehensive data ingestion and transformation pipelines developed to ensure data integrity, consistency, and readiness for model training.

### 3.1 Data Ingestion: 
The data ingestion process is responsible for acquiring raw data, performing initial cleaning, and organizing the data into structured subsets suitable for training, validation, and testing. The key steps involved are:


#### 1. Data Loading:
 
 - Source: BAF Dataset Suite Datasheet  [Jesus et al., 2022], https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data 


#### 2. Data Spliting:
The dataset is partitioned into training, validation, and testing sets based on the ```month``` feature to simulate realistic temporal splits:
 
- **Training Set**: Comprises data from months 1 to 5.
- **Validation Set**: Contains data from month 6.
- **Testing Set**: Includes data from months 7 and 8.

### 3.2 Data Transformation: 
Post-ingestion, the data undergoes a series of transformation steps to prepare for model training. These transformations address issues such as missing values, feature scaling, encoding categorical variables, and feature selection. The transformation pipeline is designed to handle different types of features appropriately. 

#### 1. Preprocessing pipelines: 

- **Numerical Features**:

    - Imputation: Missing values are addressed using the median strategy via ```SimpleImputer```, ensuring that imputations are robust to outliers.
    - Scaling: ```RobustScaler``` is employed to standardize numerical features, mitigating the impact of extreme values.
    - Feature Selection: ```SelectKBest``` with the ANOVA F-test ```(f_classif)``` is used to retain the top 16 numerical features based on their statistical significance.

- **Categorical Features**:

    - Imputation: Missing categorical values are filled using the most frequent value strategy.
    - Encoding: ```OneHotEncoder``` converts categorical variables into a binary matrix, facilitating their inclusion in the model.
    - Feature Selection: ```SelectKBest``` with the chi-squared test ```(chi2)``` selects the top 19 categorical features that are most relevant to the target variable.

- **Binary Features**:
    - Imputation: Missing values in binary features are imputed using the most frequent value.


#### 2. Saving **preprocessor** object:
All of the pipelines above will be integrated together into a **preprocessor** object.

The preprocessor project is saved as ```preprocessor.pkl``` in the artifacts directory. This ensures that the exact transformation pipeline can be reapplied during model retraining or deployment






## Model Development

1. #### Model Development

Multiple classification approaches—**Logistic Regression**, **Random Forest**, and **LightGBM**—were evaluated to detect fraudulent credit card applications, with **NearMiss** and **SMOTE** techniques (i.e., undersampling and oversampling) addressing the significant class imbalance in the dataset. Ultimately, **LightGBM** was chosen for **hyperparameter tuning** via **Optuna**, given its strong performance.

Following the model’s training phase, **calibration** (specifically **sigmoid/Platt scaling**) was applied to convert raw LightGBM scores into well-calibrated probabilities. This step ensures that each predicted score aligns more closely with the actual likelihood of fraud.

In the **final phase** of training—*after* calibration—a single numeric threshold is **fixed** based on a **business-driven** constraint (e.g., **“FPR ≤ 0.2”**). By determining this threshold via a specialized function:

```python
threshold, fpr, recall_ = find_best_threshold_for_max_recall_at_fpr(val_probs, y_val, target_fpr=0.2)
```

the model’s probability outputs are **decoupled** from frequent threshold adjustments. Consequently, the business decision boundary remains stable—even if the model is retrained or updated—eliminating the need to continually re-tune thresholds for evolving data.




## Deployment
The model is deployed using a Flask application running on AWS Elastic Beanstalk, which manages the underlying Amazon EC2 instances. An AWS CodePipeline has also been set up to automate the continuous integration and delivery (CI/CD) process. This pipeline ensures that any new code pushed to GitHub is automatically built, tested, and deployed, keeping the production environment up to date with minimal manual intervention.

## Usage

- Create new environment
    ``` 
    conda create -p venv python==3.8 -y 
    ```
- Run the environment
    ```
    conda activate venv/
    ```
- Acquire nessesary libraries
    ```
    pip install -r requirements.txt
    ```

## Git
- Revert recent commit
    ```
    git reset --soft HEAD~1
    ```
- Add heavy file to .gitattributes
    ```
    git lfs track "path/to/file"
    ```

