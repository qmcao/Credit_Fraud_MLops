import logging
import pandas as pd
import numpy as np
from src.components.data_extraction import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

DEBUG = True

def main():
    
    if DEBUG:
        # ingestion = DataIngestion()
        # train_path, test_path = ingestion.init_data_ingestion()
        # print(train_path, test_path)
        
        train_path = 'artifacts/processed/train.csv'
        test_path = 'artifacts/processed/test.csv'
        
        trainer = ModelTrainer()
        
        trainer.main(train_path, test_path)
                    
        # train,test = transformation.clean_data(train_data, test_data)
        
        # # print(cleaned_train.head())
        # # print(cleaned_test.head())
        # # print(cleaned_train.shape)
        # # print(cleaned_test.shape)
        # trainer = ModelTrainer()
        # print(type(trainer.config))
        
        # # print(trainer.models)
        # # print(trainer.params)
        # # print(trainer.metrics)
        

        # X_train, X_test, y_train, y_test = trainer.split_feature_target(train, test)
        # print(trainer.models, trainer.params)
        # model, best_params, best_cv_scores, results = trainer.train_model(np.array(X_train), np.array(y_train), trainer.models[0], trainer.params[0])
        
        # print(results, best_params)
        
        
        return 0
    
    #Load data
    ingestion = DataIngestion()
    train_path, test_path = ingestion.init_data_ingestion()
    # print(train.shape, test.shape)
    logging.info("Ingestion process completed sucessfully")
    
if __name__ == "__main__":
    main()