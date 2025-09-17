import sys
import pandas as pd
from src.utils.exception import CustomeException
from src.utils.utils import load_object
import os
import random

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomeException(e,sys)
        
        
class CustomData:
    def __init__(  self,
        income: float,
        first_name: str,
        last_name: str,
        age: int,
        bank_months_count: int,
        housing_status: str, # array(['BC', 'BE', 'BD', 'BA', 'BB', 'BF', 'BG'], dtype=object)
        current_address_months_count: int,
        proposed_credit_limit  : int,
        employment_status: str,
        email_is_free: int,
        has_other_cards: int, 
        ):
        
        self.income = income

        self.first_name = first_name

        self.last_name = last_name

        self.age = age

        self.bank_months_count = bank_months_count

        self.housing_status = housing_status

        self.current_address_months_count = current_address_months_count
        
        self.proposed_credit_limit = proposed_credit_limit
        
        self.employment_status = employment_status
        
        self.email_is_free = email_is_free
        
        self.has_other_cards = has_other_cards
        
    def get_data_as_data_frame(self):
        try:
            #Need fix
            payment_options = ['AA', 'AD', 'AB', 'AC', 'AE']
            phone_number_valid = [0, 1]
            phone_numer_weight = [0.2, 0.8]
            
            custom_data_input_dict = {
                "income": [self.income],
                "name_email_similarity": [random.uniform(0,1)], # Implement later
                "current_address_months_count": [self.current_address_months_count],
                "customer_age": [self.age],
                "days_since_request":[random.randint(0, 79)],
                "payment_type":[random.choice(payment_options)],
                "zip_count_4w":[random.randint(1,6830)],
                "velocity_6h":[random.randint(1,16818)],
                "velocity_24h":[random.randint(1297,9586)],
                "velocity_4w":[random.randint(2825, 7020)],
                "bank_branch_count_8w":[random.randint(0,2404)],
                "date_of_birth_distinct_emails_4w":[random.randint(0,39)],
                "employment_status":[self.employment_status],
                "credit_risk_score":[random.uniform(0, 389)],
                "email_is_free":[self.email_is_free],
                "housing_status": [self.housing_status],
                "phone_home_valid": [random.choices(phone_number_valid, phone_numer_weight)[0]],
                "phone_mobile_valid": [random.choices(phone_number_valid, phone_numer_weight)[0]],
                "bank_months_count": [self.bank_months_count],
                "has_other_cards": [self.has_other_cards],
                "proposed_credit_limit": [self.proposed_credit_limit], 
                "foreign_request": [0],
                "source": ["INTERNET"], 
                "session_length_in_minutes": [random.randint(0, 107)],
                "device_os": ["windows"],
                "keep_alive_session": [0],
                "device_distinct_emails_8w":[0],
                "month": [random.randint(0, 7)]

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomeException(e, sys)
        