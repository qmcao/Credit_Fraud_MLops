import os
import sys
from src.utils.exception import CustomeException
from src.utils.logger import logging
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split



class DataIngestion:
    '''
    Handles the ingestion of the dataset
    '''
    def __init__(self):
        '''Init the DataIngestion class with a configuration object'''
        self.config = self.load_config()
        self.path = self.config['source']['path']
        
        # Split config
        self.split_column: str = self.config['split']['column']
        self.train_values: list = self.config['split']['train_values']
        self.test_values: list = self.config['split']['test_values']
        
        # Sampling config
        self.sampling: bool = self.config['sampling']['enabled']
        self.sampling_type: str = self.config['sampling']['type']
        self.sampling_frac: float = self.config['sampling']['value']
        
        # ouput config
        self.out_path = self.config['output']['root']
        self.out_format = self.config['output']['format']
        self.overwrite = self.config['output']['overwrite']
        
        self.random_state = self.config['global']['random_state']
        
    def init_data_ingestion(self):
        '''
        This function read data from source, split the data into train set, test set
        by month, then save it to the artifacts/raw dir
        '''
        # Read from data source
        source = self.path
        df_full = pd.read_csv(source)
        
        # sample for small dataset
        df = df_full.loc[train_test_split(df_full.index, train_size=self.sampling_frac, stratify=df_full['fraud_bool'], 
                                          random_state=self.random_state)[0]]

        # Train test splitd base on time (months)
        col_to_split = self.split_column
        train_months = self.train_values
        test_months = self.test_values        
        train_set = df[df[col_to_split].isin(train_months)]
        test_set = df[df[col_to_split].isin(test_months)]
        
        
        # Save the dataframes as CSV files
        train_out_path = os.path.join(self.out_path, 'train.csv')
        test_out_path = os.path.join(self.out_path, 'test.csv')
        train_set.to_csv(train_out_path, index=False, header=True)
        test_set.to_csv(test_out_path, index=False, header=True)        
        
        # Return the paths to the saved CSV files
        return(
            train_out_path,
            test_out_path)
    
    # -- helper #
    def load_config(self):
        with open('conf/data.yml', 'r') as config_file:
            return yaml.safe_load(config_file)