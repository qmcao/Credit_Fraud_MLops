# Feature engineer, data cleaning

import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler

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
            numerical_features = ['Time', 'Amount']
            num_pipeline = Pipeline(
                steps=[
                    # Add to the pipeline if need to handle missing value
                    #("imputer", SimpleImputer(strategy="median")),
                    ("robust scaler", RobustScaler())
                ]
            )
            
            
            # cat_pipeline=Pipeline(

            #     steps=[
            #     ("imputer",SimpleImputer(strategy="most_frequent")),
            #     ("one_hot_encoder",OneHotEncoder()),
            #     ("scaler",StandardScaler(with_mean=False))
            #     ]

            # )
            
            logging.info(f"Numerical columns: {numerical_features}")
            preprocessor= ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                #("cat_pipelines",cat_pipeline,categorical_columns)
            ])
            
            return preprocessor
            
        except Exception as e:
            raise CustomeException(e,sys)
        
    def init_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()
            
            target_column_name= "Class"
            numerical_columns = ['Time', 'Amount']
            
            
            X_train = train_df.drop(columns=[target_column_name], axis=1)     
            y_train = train_df[target_column_name]
            
            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]
            
            logging.info(
                "Applying preprocessing on training dataframe and testing dataframe"
            )
            #Drop the unprocess columns
            tmp = X_train.drop(['Time', 'Amount'], axis=1)
            input_feature_train_arr = preprocessor_obj.fit_transform(X_train)
            
            tmp2 = X_test.drop(['Time', 'Amount'], axis=1)
            input_feature_test_arr = preprocessor_obj.transform(X_test)
            
            train_arr = np.c_[tmp, input_feature_train_arr, np.array(y_train)]
            test_arr = np.c_[tmp2, input_feature_test_arr, np.array(y_test)]
            
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )
            logging.info(f"Saved preprocessing object.")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomeException(e, sys)