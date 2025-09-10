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
        
    def encode_cat_cols(self, train_data: pd.DataFrame, test_data: pd.DataFrame, feature_list: list[str]):
        '''
        This function encode categorical variables using OneHotEncoder(sparse = False)
        '''
        # Init encoder object and fit on training set
        train = train_data[feature_list]
        test = test_data[feature_list]
        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(train_data[feature_list])
        
        # Transform on both training and test data
        train_data_encoded = pd.DataFrame(
            encoder.transform(train),
            columns= encoder.get_feature_names_out(feature_list),
            index= train.index
        )
        test_data_encoded = pd.DataFrame(
            encoder.transform(test),
            columns= encoder.get_feature_names_out(feature_list),
            index = test.index
        )
        
        return train_data_encoded, test_data_encoded
    
    
    def scale_num_cols(self, train_data: pd.DataFrame, test_data: pd.DataFrame, feature_list: list[str]):
        '''
        This function scale the numerical variables using Robust Scaler due to large number of outliers
        '''
        # Init scaler object and fit on training set
        scaler = RobustScaler()
        train = train_data[feature_list]
        test = test_data[feature_list]
        scaler.fit(train)
        
        # Transform on both training and test data
        train_data_scaled = pd.DataFrame(
            scaler.transform(train),
            columns= feature_list,
            index= train.index
        )
        test_data_scaled = pd.DataFrame(
            scaler.transform(test),
            columns= feature_list,
            index = test.index
        )
        
        return train_data_scaled, test_data_scaled
        
    
    def clean_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean train and test datasets by:
        
        1. Dropping columns with >70% missing values.
        2. Imputing remaining missing values 
            (median for numeric, most_frequent for categorical).
        3. One hot encode categorical variable
        4. Scale numerical variable

        Returns cleaned train and test DataFrames.
        """
        # Prepare what to impute
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
        #Init impute strat
        Imputer = SimpleImputer()
        numeric_strat = "median" # Highly skewed data
        cat_strat = "most_frequent"
        

        # Start cleaning process : Dropping bad columns -> Impute -> Transform (Encode + Scale)
        # Drop bad columns
        train, test = self.drop_data(train_data, test_data)     
        
        # Impute missing columns
        imputed_train_numeric, imputed_test_numeric = self.impute_data(train, test, numerical_features, Imputer, numeric_strat)
        imputed_train_cat, imputed_test_cat = self.impute_data(train, test, cat_features, Imputer, cat_strat)
        
        # Concat cleaned numerical and categorical data to apply transformation
        imputed_train = pd.concat([imputed_train_numeric, imputed_train_cat], axis=1)
        imputed_test = pd.concat([imputed_test_numeric, imputed_test_cat], axis=1)
        
        # Encoded categorical columns
        encoded_train_cat, encoded_test_cat = self.encode_cat_cols(imputed_train, imputed_test,cat_features)
        scaled_train_num, scaled_test_num = self.scale_num_cols(imputed_train, imputed_test,numerical_features)
        
        # Concat cleaned numerical and categorical data to return 
        cleaned_train = pd.concat([encoded_train_cat, scaled_train_num], axis=1)
        cleaned_test = pd.concat([encoded_test_cat, scaled_test_num], axis=1) 
        
        return cleaned_train, cleaned_test
    
    