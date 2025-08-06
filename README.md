# Credit Card Fraud Detection with MLOps

![CI/CD](https://img.shields.io/badge/CI%2FCd-AWS%20CodePipeline-blue)
![Language](https://img.shields.io/badge/Language-Python-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-green)
![deployment](https://img.shields.io/badge/Deployment-AWS%20Elastic%20Beanstalk-orange)

This project implements an end-to-end MLOps pipeline for detecting fraudulent credit card applications. It leverages advanced machine learning techniques, automated pipelines, and a decoupled architecture to build a robust and maintainable fraud detection system.

## Table of Contents
1. [Project Overview](#project-overview)
2. [MLOps Pipeline](#mlops-pipeline)
3. [Project Structure](#project-structure)
4. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
5. [How to Run](#how-to-run)
    - [Training Pipeline](#training-pipeline)
    - [Prediction API](#prediction-api)
6. [Model Development and Decoupling](#model-development-and-decoupling)
7. [Deployment](#deployment)
8. [Future Work](#future-work)
9. [Contributing](#contributing)
10. [License](#license)
11. [Contact](#contact)

## Project Overview

Credit card fraud is a significant threat in the financial industry. Traditional fraud detection models often struggle to keep up with evolving fraud patterns and require frequent, time-consuming threshold adjustments. This project addresses these challenges by implementing a decoupled architecture, as proposed by Luzio et al. (2024). By calibrating model outputs and using a fixed business threshold, the system can adapt to new data while maintaining consistent decision criteria.

The primary objectives of this project are:
- **Develop a highly accurate fraud detection model** using LightGBM with hyperparameter tuning (Optuna) and calibration.
- **Build an automated MLOps pipeline** for data ingestion, transformation, and model training.
- **Implement a decoupling strategy** to separate the ML model from the business threshold, ensuring stability and reducing maintenance overhead.
- **Deploy the model as a web service** with a CI/CD pipeline for continuous delivery.

## MLOps Pipeline

The project follows a structured MLOps approach to automate the machine learning lifecycle.

![Project MLOps Diagram](project-mlops-diagram.svg)

The pipeline consists of the following key stages:
1.  **Data Ingestion**: Raw data is ingested and split into training, validation, and test sets based on time-series to simulate a real-world scenario.
2.  **Data Transformation**: A preprocessing pipeline handles missing values, scales numerical features, encodes categorical variables, and performs feature selection. The `preprocessor` object is saved for consistent application during training and prediction.
3.  **Model Training**: The LightGBM model is trained with optimized hyperparameters. The model's output is calibrated to produce reliable probabilities.
4.  **Model Decoupling**: A fixed business threshold is determined based on the desired False Positive Rate (FPR). This decouples the model from the business logic, allowing the model to be updated without changing the threshold.
5.  **Deployment**: The trained model and preprocessing pipeline are deployed as a Flask API on AWS Elastic Beanstalk, with a CI/CD pipeline set up using AWS CodePipeline for automated deployments.

## Project Structure
```
Credit_Fraud_MLops/
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── ...
├── notebook/
│   ├── 1. EDA FRAUD APPLICATION .ipynb
│   ├── 2. MODEL TRAINING.IPYNB
│   └── 3. MODEL PERFORMANCE MONITERING..ipynb
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── __init__.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── templates/
│   └── home.html
├── application.py
├── Dockerfile
├── requirements.txt
└── setup.py
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Conda for environment management

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Credit_Fraud_MLops.git
    cd Credit_Fraud_MLops
    ```
2.  **Create and activate the conda environment:**
    ```bash
    conda create -p venv python==3.8 -y
    conda activate ./venv
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### Training Pipeline
To run the full training pipeline, which includes data ingestion, transformation, and model training:
```bash
python -m src.pipeline.train_pipeline
```
The trained artifacts (`model.pkl`, `preprocessor.pkl`) will be saved in the `artifacts/` directory.

### Prediction API
To start the Flask server for predictions:
```bash
python application.py
```
The application will be available at `http://127.0.0.1:5000`. You can send POST requests to the `/predict` endpoint with JSON data to get fraud predictions.

## Model Development and Decoupling

The model development process involved evaluating several classifiers (Logistic Regression, Random Forest, LightGBM) and handling class imbalance with SMOTE and NearMiss. LightGBM with Optuna for hyperparameter tuning was selected for its performance.

A key innovation in this project is the **decoupling** of the model from the business threshold. After training and calibration (using sigmoid/Platt scaling), a single threshold is fixed based on a business constraint (e.g., `FPR <= 0.2`). This is achieved with the following function:
```python
threshold, fpr, recall_ = find_best_threshold_for_max_recall_at_fpr(val_probs, y_val, target_fpr=0.2)
```
This approach ensures that the business decision boundary remains stable even when the model is retrained, eliminating the need for constant threshold adjustments.

## Deployment

The model is deployed as a Flask application on **AWS Elastic Beanstalk**. A **CI/CD pipeline** using **AWS CodePipeline** is configured to automatically build, test, and deploy new versions of the application whenever changes are pushed to the GitHub repository.

## Future Work
- Implement a monitoring system to detect model drift and trigger retraining.
- Explore more advanced feature engineering techniques.
- Expand the API to provide more detailed explanations for fraud predictions.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
Minh Cao - [Your Email] - [Your LinkedIn Profile]

