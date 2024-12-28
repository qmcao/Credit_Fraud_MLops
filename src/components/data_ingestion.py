import os
import sys
from src.exception import CustomeException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig

# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer

RANDOM_STATE = 42
@dataclass
class DataIngestionConfig:
    '''
    Object to store the path of data
    '''
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
    
    def init_data_ingestion(self):
        #read from different sources
        logging.info("Begin data ingestion method")
        try:
            df = pd.read_csv('notebook\data\creditcard.csv')
            logging.info('Read csv file as dataframe')
            
            #Create dir for data_path
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Init train test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state= RANDOM_STATE)
            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data ingestion is completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomeException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    obj.init_data_ingestion()