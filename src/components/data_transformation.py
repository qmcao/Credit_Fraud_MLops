import numpy as np 
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from src.utils.logger import logging
import os
import joblib


# Use to produce preprocessor obj
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataTransformation:
    def __init__(self):
        '''
        Init data transformation config for the class
        '''
        self.config = self.load_config()

        # Features
        self.res_col = self.config['features']['response_col']
        self.bad_cols = self.config['features']['bad_cols']
        self.missing_thres = self.config['features']['missing_thres']
        self.num_features = self.config['features']['num_cols']
        self.cat_features = self.config['features']['cat_cols']
        
        # Strategy
        self.scaling_strat = self.config['strategy']['scaler']
        self.num_impute_strat = self.config['strategy']['impute']['num_strat']
        self.cat_impute_strat = self.config['strategy']['impute']['cat_strat']
        self.encode_unknown = self.config['strategy']['encode']['unknow_strat']
        self.encode_sparse = self.config['strategy']['encode']['sparse_output']
        
        # output_path
        self.out_path = self.config['output']['root']
        
    # --- step ---

    def drop_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identifies columns in a dataframe with more than 70% missing values.

        Drop the corresponding columns from the data
        """
        
        # Convert negative value to nan and calculate the proportion of missing values for each column
        columns_to_check = self.bad_cols
        threshold = self.missing_thres
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
        encoder = OneHotEncoder(sparse_output=self.encode_sparse, handle_unknown=self.encode_unknown)
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
        scaler = None
        if self.scaling_strat == 'robust-scaler':
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
        
    
    def transform_data(self, train_path: str, test_path: str):
        """
        Function to orchestrate the data transformation process
        
        Transformed train and test datasets by:
        
        1. Dropping columns with >70% missing values.
        2. Imputing remaining missing values 
            (median for numeric, most_frequent for categorical).
        3. One hot encode categorical variable
        4. Scale numerical variable

        Returns cleaned train and test DataFrames.
        """
        # Read from artifacts folder
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # Get y_train, y_test
        y_train, y_test = self.get_response_column(train_data, test_data)
        
        # Start cleaning process : Dropping bad columns -> Impute -> Transform (Encode + Scale)
        # Drop bad columns
        train, test = self.drop_data(train_data, test_data)     
        
        # Impute missing columns
        train_num, test_num = self.impute_data(train, test, self.num_features, self.num_impute_strat)
        train_cat, test_cat = self.impute_data(train, test, self.cat_features, self.cat_impute_strat)
        
        # Encoded categorical columns, scale numerical
        encoded_train_cat, encoded_test_cat = self.encode_cat_cols(train_cat, test_cat,self.cat_features)
        scaled_train_num, scaled_test_num = self.scale_num_cols(train_num, test_num, self.num_features)
        
        # Concat cleaned numerical and categorical data to return 
        transformed_train = pd.concat([encoded_train_cat, scaled_train_num, y_train], axis=1)
        transformed_test = pd.concat([encoded_test_cat, scaled_test_num, y_test], axis=1) 
        
        # Save the dataframes as CSV files in the artifacts/processed folder
        os.makedirs(self.out_path, exist_ok=True) # create folder if not exist
        transformed_train_out_path = os.path.join(self.out_path, 'train.csv')
        transformed_test_out_path = os.path.join(self.out_path, 'test.csv')
        transformed_train.to_csv(transformed_train_out_path, index=False, header=True)
        transformed_test.to_csv(transformed_test_out_path, index=False, header=True)      
        
        return (transformed_train_out_path, 
                transformed_test_out_path)

    
    
    
    
    # -- helper --
    def load_config(self):
        with open('conf/features.yml', 'r') as config_file:
            return yaml.safe_load(config_file)
        
        
    def get_response_column(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Return the response variable with its original index
        """
        res_col = self.res_col
        return train_data[res_col], test_data[res_col]
    
    
    def build_preprocessor(self):
        """
        Build a scikit-learn preprocessor (ColumnTransformer) based on the config.
        This preprocessor can be used in a model pipeline.
        """
        # ---- numerical pipeline ----
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=self.num_impute_strat)),
            ("scaler", RobustScaler() if self.scaling_strat == "robust-scaler" else StandardScaler())
        ])

        # ---- categorical pipeline ----
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=self.cat_impute_strat)),
            ("encoder", OneHotEncoder(
                handle_unknown=self.encode_unknown,
                sparse_output=self.encode_sparse
            ))
        ])

        # ---- combine ----
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.num_features),
                ("cat", cat_pipeline, self.cat_features)
            ]
        )
        return preprocessor    
    
    def save_preprocessor(self, preprocessor, filename="preprocessor.pkl"):
        """
        Save the preprocessor object into the artifacts folder.
        """
        os.makedirs(self.out_path, exist_ok=True)
        preprocessor_path = os.path.join(self.out_path, filename)
        joblib.dump(preprocessor, preprocessor_path)
        return preprocessor_path