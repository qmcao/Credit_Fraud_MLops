import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from dataclasses import dataclass, field
from src.utils.logger import logging



@dataclass
class DataTransformationConfig:
    # Missing threshold and features
    drop_missing_threshold: float = 0.70
    negative_to_nan_cols: list[str] = field(default_factory=lambda:[
        "prev_address_months_count", "current_address_months_count",
        "intended_balcon_amount", "bank_months_count",
        "session_length_in_minutes", "device_distinct_emails_8w",
    ])
    
    # Feature groups
    numerical_features: list[str] = field(default_factory=lambda:[
        "income", "name_email_similarity", "current_address_months_count",
        "customer_age", "days_since_request", "zip_count_4w", "velocity_6h",
        "velocity_24h", "velocity_4w", "bank_branch_count_8w",
        "date_of_birth_distinct_emails_4w", "credit_risk_score",
        "bank_months_count", "proposed_credit_limit",
        "session_length_in_minutes", "device_distinct_emails_8w", "month",
    ])
    categorical_features: list[str] = field(default_factory=lambda: [
        "payment_type", "employment_status", "housing_status", "source",
        "device_os", "email_is_free", "phone_home_valid",
        "phone_mobile_valid", "has_other_cards", "foreign_request",
        "keep_alive_session",
    ])
    
     # Impute strategies
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    
    # Enc/Scale params
    onehot_handle_unknown: str = "ignore"   # important for inference robustness
    onehot_sparse_output: bool = False
    scaler: RobustScaler = RobustScaler() 
    
    # response column
    response_column: str = "fraud_bool"
       

class DataTransformation:
    def __init__(self):
        '''
        Init data transformation config for the class
        '''
        self.transformation_config = DataTransformationConfig()
        
    # --- step ---
    def get_response_column(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Return the response variable with its original index
        """
        res_col = self.transformation_config.response_column
        
        return train_data[res_col], test_data[res_col]
        
        
    def drop_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identifies columns in a dataframe with more than 70% missing values.

        Drop the corresponding columns from the data
        """
        
        # Convert negative value to nan and calculate the proportion of missing values for each column
        columns_to_check = self.transformation_config.negative_to_nan_cols
        threshold = self.transformation_config.drop_missing_threshold
        train_data[columns_to_check] = train_data[columns_to_check].where(train_data[columns_to_check] >= 0, np.nan)
        missing_ratio = train_data.isnull().sum() / len(train_data)        
        columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        
        # Drop the corresponding columns
        new_train_data = train_data.drop(columns_to_drop, axis = 1)
        new_test_data = test_data.drop(columns_to_drop, axis = 1)

        return new_train_data, new_test_data

    def impute_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                    feature_list: list[str], strat:str ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        This function handle impute logic
        """
        
        #Fit on training data then transform on train and test data to prevent leakage
        imputer = SimpleImputer(strategy=strat)
        imputer.fit(train_data[feature_list])

        #Transform training data
        train_data_imputed = pd.DataFrame(
            imputer.transform(train_data[feature_list]),
            columns=feature_list,
            index= train_data.index
        )
        
        # Transform testing data
        test_data_imputed = pd.DataFrame(
            imputer.transform(test_data[feature_list]),
            columns=feature_list,
            index= test_data.index
        )
        
        return train_data_imputed, test_data_imputed
        
    def encode_cat_cols(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                        feature_list: list[str])-> tuple[pd.DataFrame, pd.DataFrame]: 
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
    
    
    def scale_num_cols(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                       feature_list: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        This function scale the numerical variables using Robust Scaler due to large number of outliers
        '''
        # Init scaler object and fit on training set
        scaler = self.transformation_config.scaler
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
        
    
    def clean_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean train and test datasets by:
        
        1. Dropping columns with >70% missing values.
        2. Imputing remaining missing values 
            (median for numeric, most_frequent for categorical).
        3. One hot encode categorical variable
        4. Scale numerical variable

        Returns cleaned train and test DataFrames.
        """
        # Get y_train, y_test
        y_train, y_test = self.get_response_column(train_data, test_data)
        
        # Start cleaning process : Dropping bad columns -> Impute -> Transform (Encode + Scale)
        # Drop bad columns
        train, test = self.drop_data(train_data, test_data)     
        
        # Impute missing columns
        train_num, test_num = self.impute_data(train, test, self.transformation_config.numerical_features,
                                                                       self.transformation_config.numeric_impute_strategy)
        train_cat, test_cat = self.impute_data(train, test, self.transformation_config.categorical_features,
                                                                        self.transformation_config.categorical_impute_strategy)
        
        # Encoded categorical columns, scale numerical
        encoded_train_cat, encoded_test_cat = self.encode_cat_cols(train_cat, test_cat,self.transformation_config.categorical_features)
        scaled_train_num, scaled_test_num = self.scale_num_cols(train_num, test_num, self.transformation_config.numerical_features)
        
        # Concat cleaned numerical and categorical data to return 
        cleaned_train = pd.concat([encoded_train_cat, scaled_train_num, y_train], axis=1)
        cleaned_test = pd.concat([encoded_test_cat, scaled_test_num, y_test], axis=1) 
        
        return cleaned_train, cleaned_test
    
    