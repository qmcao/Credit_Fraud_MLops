import os
import sys
from src.exception import CustomeException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

RANDOM_STATE = 42
@dataclass
class DataIngestionConfig:
    '''
    This dataclass holds the file paths for the output artifacts of the data
    ingestion process, including the raw, training, and test datasets.
    '''
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")

class DataIngestion:
    '''
    Handles the ingestion of the dataset
    '''
    def __init__(self):
        '''Init the DataIngestion class with a configuration object'''
        self.ingestion_config= DataIngestionConfig()
    
    def init_data_ingestion(self):
        '''
        Orchestrates the data ingestion process
        '''
        # Read from data source
        df = pd.read_csv("s3://ml-feature-store-029552764749/credit_fraud/data/fraud_1.csv")
        
        # Train test splitd base on time (months)
        train_months = [1, 2, 3, 4, 5, 6]
        test_months = [7, 8]        
        train_set = df[df["month"].isin(train_months)]
        test_set = df[df['month'].isin(test_months)]
        
        # Save the dataframes as CSV files
        train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)        
        
        # Return the paths to the saved CSV files
        return(
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path)
    
    
    
    