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
    
    
    def drop_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies columns in a dataframe with more than 70% missing values.

        Drop the corresponding columns from the data
        """
        threshold = 0.7
        columns_to_check = ['prev_address_months_count', 'current_address_months_count', 'intended_balcon_amount', 'bank_months_count', 
                    'session_length_in_minutes', 'device_distinct_emails_8w']  
        
        # Convert negative value to nan and calculate the proportion of missing values for each column
        train_data[columns_to_check] = train_data[columns_to_check].where(train_data[columns_to_check] >= 0, np.nan)
        missing_ratio = train_data.isnull().sum() / len(train_data)        
        columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        
        # Drop the corresponding columns
        new_train_data = train_data.copy()
        new_test_data = test_data.copy()
        new_train_data.drop(columns_to_drop, axis = 1, inplace = True)
        new_test_data.drop(columns_to_drop, axis = 1, inplace = True)
        
        return new_train_data, new_test_data

    def impute_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame, feature_list: list[str], 
                    imputer:SimpleImputer, strat:str ) ->pd.DataFrame:
        """
        This function handle impute logic
        """
        
        #Fit on training data then transform on train and test data to prevent leakage
        Imputer = imputer.set_params(strategy = strat)
        Imputer.fit(train_data[feature_list])

        #Transform training data
        train_data_imputed = pd.DataFrame(
            Imputer.transform(train_data[feature_list]),
            columns=feature_list,
            index= train_data.index
        )
        
        # Transform testing data
        test_data_imputed = pd.DataFrame(
            Imputer.transform(test_data[feature_list]),
            columns=feature_list,
            index= test_data.index
        )
        
        return train_data_imputed, test_data_imputed
        
    def clean_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean train and test datasets by:
        
        1. Dropping columns with >70% missing values.
        2. Imputing remaining missing values 
            (median for numeric, most_frequent for categorical).
        3. (Future) Apply feature transformations.

        Returns cleaned train and test DataFrames.
        """
        # Prepare what to impute and impute strategy
        numerical_features = ['income', 'name_email_similarity', 'current_address_months_count', 'customer_age', 'days_since_request'
        , 'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w', 'bank_branch_count_8w', 
        'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'bank_months_count', 'proposed_credit_limit',  'session_length_in_minutes',
        'device_distinct_emails_8w', 'month']
        
        cat_features = ['payment_type', 'employment_status', 'housing_status',
                                'source', 'device_os', 'email_is_free',
                                    'phone_home_valid',
                                    'phone_mobile_valid',
                                    'has_other_cards',
                                    'foreign_request',
                                    'keep_alive_session']
        
        Imputer = SimpleImputer()
        numeric_strat = "median" # Highly skewed data
        cat_strat = "most_frequent"
        
        
        # Start cleaning process : Dropping bad columns -> Impute -> Transform
        train, test = self.drop_data(train_data, test_data)     
        imputed_train_numeric, imputed_test_numeric = self.impute_data(train, test,
                                                                 numerical_features, Imputer, numeric_strat)
        imputed_train_cat, imputed_test_cat = self.impute_data(train, test,
                                                                 cat_features, Imputer, cat_strat)
        
        # Concat cleaned numerical and categorical data together to return
        cleaned_train = pd.concat([imputed_train_numeric, imputed_train_cat], axis=1)
        cleaned_test = pd.concat([imputed_test_numeric, imputed_test_cat], axis=1)
        
        return cleaned_train, cleaned_test
    
    def get_data_transformer_object(self, train: pd.DataFrame, test: pd.DataFrame):
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
        
    def init_data_transformation(self, train_path, val_path, test_path):
        '''
        @return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test and path to preprocessor.pkl file
        '''
        try:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train, validation and test data completed")
            
            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()
            
            target_column_name= "fraud_bool"
            
            X_train = train_df.drop(columns=[target_column_name], axis=1)     
            y_train = train_df[target_column_name]
            
            X_val = val_df.drop(columns=target_column_name, axis=1)
            y_val = val_df[target_column_name]
            
            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]
            
            logging.info(
                "Applying preprocessing on training dataframe, validation dataframe and testing dataframe"
            )
            
            X_train_processed = preprocessor_obj.fit_transform(X_train, y_train)
            X_val_processed = preprocessor_obj.transform(X_val)
            X_test_processed = preprocessor_obj.transform(X_test)     
           
            logging.info("Preprocessing completed.")
            
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )
            logging.info(f"Saved preprocessing object.")
            
            return (
                X_train_processed,
                X_val_processed,
                X_test_processed,
                np.array(y_train),
                np.array(y_val),
                np.array(y_test),
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomeException(e, sys)