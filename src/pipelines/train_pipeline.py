import logging
import pandas as pd
import numpy as np
from src.components.data_extraction import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

DEBUG = False

def main():
    if DEBUG == False:
        
        #Ingestion process
        logging.info("Init Ingestion process")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.init_data_ingestion()
        logging.info("Ingestion process completed sucessfully")
        
        # Data transformation process
        logging.info("Init Data transformation process")
        transform = DataTransformation()
        processed_train_path, processed_test_path = transform.transform_data(train_path, test_path)
        preprocessor = transform.build_preprocessor()
        _ = transform.save_preprocessor(preprocessor)
        
        logging.info("Data transformation process complete")
        
        # Model training process
        logging.info("Init Model training process")
        model_trainer = ModelTrainer()
        model_file_path, metrics_file_path = model_trainer.main(processed_train_path, processed_test_path)
        logging.info("Model training process complete")
        
        

if __name__ == "__main__":
    main()