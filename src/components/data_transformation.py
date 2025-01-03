# Feature engineer, data cleaning

import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2

from src.exception import CustomeException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    '''
    Object storing the preprocessor.pkl file path
    '''
    preprocessor_obj_file_path= os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function is used for creating data transformation object (e.g preprocessor)
        '''
        try:
            #Numerical pipeline for data transformation
            numerical_features = ['income', 'name_email_similarity', 'current_address_months_count', 'customer_age', 'days_since_request'
                    , 'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w', 'bank_branch_count_8w', 
                    'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'bank_months_count', 'proposed_credit_limit',  'session_length_in_minutes',
                    'device_distinct_emails_8w', 'month']
            
            num_pipeline = Pipeline(
                steps=[
                    # Add to the pipeline if need to handle missing value
                    ("imputer", SimpleImputer(strategy="median")),
                    ("robust scaler", RobustScaler()),
                    ('num_feature_select', SelectKBest(score_func=f_classif, k=16)) # According to EDA
                ]
            )
            
            #Categorical pipeline for data transformation
            categorical_features = ['payment_type', 'employment_status', 'housing_status',
                         'source', 'device_os']
            
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(sparse=False)),
                ('cat_feature_select', SelectKBest(score_func=chi2, k=19)), # According to EDA
                #("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            # --- Binary Pipeline ------
            # -----------------------------------------------
            # Usually, keep binary features as 0/1 with minimal or no transformation.
            # - Optional imputation if these fields have missing values
            # - Typically no scaling or OHE needed
            binary_features = [
                'email_is_free',
                'phone_home_valid',
                'phone_mobile_valid',
                'has_other_cards',
                'foreign_request',
                'keep_alive_session',
            ]

            binary_pipeline = Pipeline([
                ('bin_imputer', SimpleImputer(strategy='most_frequent')),
            ])
            
            logging.info(f"Numerical columns: {numerical_features}")
            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Binary columns: {binary_features}")
            
            preprocessor= ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline",cat_pipeline, categorical_features),
                ("bin_pipeline", binary_pipeline, binary_features)
            ])
            
            logging.info("Preprocessor object created successfully.")

            return preprocessor
            
        except Exception as e:
            raise CustomeException(e,sys)
        
    def init_data_transformation(self, train_path, test_path):
        '''
        @return X_train_processed, X_test_processed, y_train, y_test and path to preprocessor.pkl file
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()
            
            target_column_name= "fraud_bool"
            
            X_train = train_df.drop(columns=[target_column_name], axis=1)     
            y_train = train_df[target_column_name]
            
            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]
            
            logging.info(
                "Applying preprocessing on training dataframe and testing dataframe"
            )
            
            X_train_preprocessed = preprocessor_obj.fit_transform(X_train, y_train)
            X_test_preprocessed = preprocessor_obj.transform(X_test)     
           
            
            # numerical_features = ['income', 'name_email_similarity', 'current_address_months_count', 'customer_age', 'days_since_request'
            #         , 'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w', 'bank_branch_count_8w', 
            #         'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'bank_months_count', 'proposed_credit_limit',  'session_length_in_minutes',
            #         'device_distinct_emails_8w', 'month']
            
            # binary_features = [
            #     'email_is_free',
            #     'phone_home_valid',
            #     'phone_mobile_valid',
            #     'has_other_cards',
            #     'foreign_request',
            #     'keep_alive_session',
            
            # ]
            # categorical_features = ['payment_type', 'employment_status', 'housing_status',
            #              'source', 'device_os']

            # # Retrieve feature names from preprocessor
            # num_features = preprocessor_obj.named_transformers_['num_pipeline'].named_steps['num_feature_select'].get_feature_names_out(numerical_features)
            # cat_features = preprocessor_obj.named_transformers_['cat_pipeline'].named_steps['cat_feature_select'].get_feature_names_out(categorical_features)
            # bin_features = preprocessor_obj.named_transformers_['bin_pipeline'].named_steps['bin_feature_select'].get_feature_names_out(binary_features)
            # # Combine all feature names
            # all_features = np.concatenate([num_features, cat_features, bin_features])
            
            # X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=all_features, index=X_train.index)
            # X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=all_features, index=X_test.index)
            logging.info("Preprocessing completed.")
            
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )
            logging.info(f"Saved preprocessing object.")
            
            return (
                X_train_preprocessed,
                X_test_preprocessed,
                np.array(y_train),
                np.array(y_test),
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomeException(e, sys)