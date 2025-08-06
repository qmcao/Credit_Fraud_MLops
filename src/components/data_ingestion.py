import os
import sys
from src.exception import CustomeException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

RANDOM_STATE = 42
@dataclass
class DataIngestionConfig:
    '''
    Object to store the path of data
    '''
    train_data_path: str=os.path.join('artifacts',"train.csv")
    val_data_path: str=os.path.join('artifacts',"val.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    '''
    Load data from source and save train, test data to path
    
    '''
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
    
    def init_data_ingestion(self):
        #read from different sources
        logging.info("Begin data ingestion method")
        try:
            df = pd.read_csv('notebook/data/fraud.csv')
            logging.info('Read csv file as dataframe')
            
            #drop intended_balcon_amount and prev_address_months_count

            df.drop(['prev_address_months_count', 'intended_balcon_amount', 'device_fraud_count'], axis=1, inplace=True)
            
            #Create dir for data_path
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Define train, val, test index 
            train_months = [1, 2, 3, 4, 5]
            val_month = [6]
            test_months = [7, 8]
            
            logging.info("Init train test split")
            train_set = df[df["month"].isin(train_months)]
            val_set = df[df['month'].isin(val_month)]
            test_set = df[df['month'].isin(test_months)]
            
            #train_set, test_set = train_test_split(df, test_size=0.2, random_state= RANDOM_STATE)
            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header=True)
            val_set.to_csv(self.ingestion_config.val_data_path, index= False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data ingestion is completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomeException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_path, val_path, test_path = obj.init_data_ingestion()
    
    data_transformation_obj = DataTransformation()
    
    X_train, X_val, X_test, y_train, y_val, y_test,_ = data_transformation_obj.init_data_transformation(train_path, val_path, test_path)
    model_trainer = ModelTrainer()
    _,_ = model_trainer.init_model_trainer(X_train, X_val, X_test, y_train, y_val, y_test)
    
    
    
    