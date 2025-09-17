import logging
import pandas as pd
import numpy as np
from src.components.data_extraction import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

DEBUG = True

def main():
    
    if DEBUG:
        ingestion = DataIngestion()
        train_path, test_path = ingestion.init_data_ingestion()
        print(train_path, test_path)
        
        
        # transformation = DataTransformation()
        # train_data = pd.read_csv("artifacts/train.csv")
        # test_data = pd.read_csv("artifacts/test.csv")


        # print(train_data['fraud_bool'])
        # cat_features = ['payment_type', 'employment_status', 'housing_status',
        #                 'source', 'device_os', 'email_is_free',
        #                     'phone_home_valid',
        #                     'phone_mobile_valid',
        #                     'has_other_cards',
        #                     'foreign_request',
        #                     'keep_alive_session']

        # numerical_features = ['income', 'name_email_similarity', 'current_address_months_count', 'customer_age', 'days_since_request'
        # , 'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w', 'bank_branch_count_8w', 
        # 'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'bank_months_count', 'proposed_credit_limit',  'session_length_in_minutes',
        # 'device_distinct_emails_8w', 'month']
        
        
        # train_data_scaled, test_data_scaled = transformation.scale_num_cols(train_data,
        #                                                                        test_data,
        #                                                                        numerical_features)
        
        # print(train_data_scaled.shape, test_data_scaled.shape)
        # print(train_data_scaled)
        # print(test_data_scaled)
        
                    
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