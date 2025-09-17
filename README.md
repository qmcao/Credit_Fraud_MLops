# Credit Card Fraud Detection with MLOps

![Language](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Containerization](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![Deployment](https://img.shields.io/badge/Amazon_Web_Services-FF9900?style=for-the-badge&logo=amazonwebservices&logoColor=white)
![CI/CD](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)


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

Credit fraud is a major challenge in the financial industry, where traditional rule-based systems demand constant manual updates and often struggle to keep pace with evolving fraud patterns. This project demonstrates an end-to-end machine learning workflow for fraud detection, including data ingestion, feature engineering, model training, and deployment as a cloud-based web service. The objective is to automate the fraud detection process, reducing manual intervention while improving adaptability. This work was inspired by my internship experience on a credit fraud team, where I explored more efficient and scalable approaches to combating fraud.

The primary goals of this project are:
- **Develop a robust fraud detection model** by benchmarking multiple machine learning algorithms and applying hyperparameter tuning to maximize performance.
- **Implement an automated MLOps pipeline** that handles data ingestion, preprocessing, feature engineering, and model training with minimal manual intervention.
- **Deploy the trained model as a cloud-based** web service using containerization and a CI/CD pipeline to enable seamless integration and continuous delivery.

## MLOps Pipeline

The project follows a structured MLOps approach to automate the machine learning lifecycle.

![Project MLOps Diagram](project-mlops-diagram.svg)

The pipeline consists of the following key stages:
1.  **Data Ingestion**: Raw data is ingested and split into training, validation, and test sets based on time-series to simulate a real-world scenario.
2.  **Data Transformation**: A preprocessing pipeline handles missing values, scales numerical features, encodes categorical variables, and performs feature selection. The `preprocessor` object is saved for consistent application during training and prediction.
3.  **Model Training**: The LightGBM model is trained with optimized hyperparameters. The model's output is calibrated to produce reliable probabilities.
4.  **Model Decoupling**: A fixed business threshold is determined based on the desired False Positive Rate (FPR). This decouples the model from the business logic, allowing the model to be updated without changing the threshold.
5.  **Deployment**: The trained model and preprocessing pipeline are containerized using Docker and deployed via a CI/CD pipeline for automated deployments.

## Project Structure
```
mlops-poc/
│
├── conf/                          # Configs (no hardcoding in code)
│   ├── data.yaml                  # data source, sample size, paths
│   ├── features.yaml              # feature selection / transformations
│   └── train.yaml                 # training params (model, hyperparams)
│
├── src/
│   ├── components/                #  
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_training.py
│   │   ├── model_evaluator.py     # (optional) eval logic
│   │   └── __init__.py
│   │
│   ├── pipelines/                 # Orchestrators (link components)
│   │   ├── train_pipeline.py      # main orchestrator (end-to-end run)
│   │   └── retrain_trigger.py     # (future) schedule/trigger script
│   │
│   └── utils/                     # Shared helpers
│       ├── utils.py               # load/save data, artifacts
│       ├── logger.py              # standard logging setup
│       └── mlflow_utils.py        # wrappers for experiment tracking
│
├── artifacts/                     # Local outputs (replace w/ S3 later)
│   ├── raw/                       # ingested data
│   ├── processed/                 # transformed data
│   ├── models/                    # trained models
│   ├── metrics/                   # evaluation metrics
│   └── logs/                      # logs, checkpoints
│
├── tests/                         # Testing
│   ├── test_components.py         # unit tests for ingestion/transform
│   └── test_pipeline_smoke.py     # integration test (tiny dataset)
│
├── docker/                        # container setup
│   └── Dockerfile                 # single image for running pipeline
│
├── requirements.txt               # Python deps
├── Makefile                       # handy shortcuts (make train, make test)
├── README.md                      # project overview + how to run
└── .gitignore

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

The model is deployed as a Docker container managed by a **GitHub Actions CI/CD pipeline**. The pipeline automates the building, testing, and deployment of the application onto a self-hosted runner (e.g., an AWS EC2 instance).

The workflow is defined in `.github/workflows/main.yml` and consists of three main jobs:

1.  **Continuous Integration (`integration`)**:
    *   Triggered on every push to the `main` branch.
    *   Performs initial checks like linting and running unit tests to ensure code quality.

2.  **Continuous Delivery (`build-and-push-ecr-image`)**:
    *   After successful integration, this job builds a Docker image of the application.
    *   It then tags the image and pushes it to a private **Amazon Elastic Container Registry (ECR)** repository.

3.  **Continuous Deployment (`Continuous-Deployment`)**:
    *   This job runs on a **self-hosted runner** (e.g., an AWS EC2 instance).
    *   It pulls the latest Docker image from ECR.
    *   It stops and removes any old running versions of the application container.
    *   Finally, it runs the new Docker container, exposing the Flask application on port 8080 to serve prediction requests.

This setup ensures that any changes merged into the main branch are automatically tested, packaged, and deployed, providing a true end-to-end automated workflow.

## Future Work
- Implement a monitoring system to detect model drift and trigger retraining.
- Explore more advanced feature engineering techniques.
- Expand the API to provide more detailed explanations for fraud predictions.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License


## Contact
Minh Cao - [qmcao@uci.edu] - [[LinkedIn](https://www.linkedin.com/in/minhcao-uci/)]

