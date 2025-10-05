# Machine Learning Operations for Credit Fraud Detection

![Language](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Containerization](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![Deployment](https://img.shields.io/badge/Amazon_Web_Services-FF9900?style=for-the-badge&logo=amazonwebservices&logoColor=white)
![CI/CD](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)


## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Pipeline / Workflow](#work-flow)
4. [How to Run](#how-to-run)
5. [Demo](#demo)
6. [Future Improvements](#future-work)
7. [Author, Contacts](#contact)


## 1. Project Overview

**Credit fraud** remains a major challenge in the financial industry, requiring constant updates to keep detection systems effective against evolving fraud patterns.

This project aims to develop a **proof-of-concept Machine Learning Operations (MLOps) pipeline** that automates and streamlines the credit fraud detection process, reducing manual intervention and improving model adaptability over time.

## 2. Project Structure
```
mlops-root/
│
├── conf/                          # Configs 
│   ├── data.yaml                  # data source, sample size, paths
│   ├── features.yaml              # feature selection / transformations
│   └── config.yaml                 # training params (model, hyperparams)
│
├── src/
│   ├── components/                # components in automated ML pipeline
│   │   ├── data_extraction.py
│   │   ├── data_transformation.py
│   │   ├── model_training.py
│   │   ├── model_validation.py    
│   │   └── __init__.py
│   │
│   ├── pipelines/                 # Orchestrators (link components)
│   │   ├── train_pipeline.py      # main orchestrator (automated ML pipeline)
|   |   ├── prediction_pipeline.py # Handle prediction process for the web app
│   │   └── retrain_trigger.py     # schedule/trigger script (future)
│   │
│   └── utils/                     # helpers
│       ├── utils.py               # load/save data, artifacts
│       ├── logger.py              # standard logging setup
│       └── mlflow_utils.py        # wrappers for experiment tracking (future)
│
├── artifacts/                     # Local outputs (replace w/ S3 later)
│   ├── raw/                       # ingested data
│   ├── processed/                 # transformed data, preprocessor object
│   ├── models/                    # trained models
│   ├── metrics/                   # evaluation metrics
│   └── logs/                      # logs, checkpoints
│
├── tests/                         # Testing (future)
│   ├── test_components.py         # unit tests for ingestion/transform
│   └── test_pipeline_smoke.py     # integration test (tiny dataset)
│
├── Dockerfile                     # container setup  
├── requirements.txt               # Python deps
├── Makefile                       # handy shortcuts (make train, make test)
├── README.md                      # project overview + how to run
└── .gitignore

```

## 3. Pipeline and Workflows

![My Diagram](diagrams/credit-mlops.drawio.svg)




## 4. How to Run

#### Prerequisites
- Python 3.8 or higher
- Conda for environment management

#### Installation
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



#### Training Pipeline
To run the full training pipeline, which includes data extraction, transformation, and model training:
```bash
python -m src.pipelines.train_pipeline
```
The trained artifacts (`model.pkl`, `preprocessor.pkl`, `train.csv`, `test.csv`, `metrics.json`) will be saved in the `artifacts/` directory.

#### Prediction API
To start the Flask server for predictions:
```bash
python application.py
```
The application will be available at `http://127.0.0.1:8080`. You can send POST requests to the `/predict` endpoint with JSON data to get fraud predictions.


## 5. Demo

## 6. Future Improvements
- Implement a monitoring system to detect model drift and trigger retraining.
- Explore more advanced feature engineering techniques.
- Expand the API to provide more detailed explanations for fraud predictions.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License


## Contact
Minh Cao - [qmcao@uci.edu] - [[LinkedIn](https://www.linkedin.com/in/minhcao-uci/)]

