import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomeException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomeException(e, sys)
    
    
def eval_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = list(params.values())[i]
            
            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_score = r2_score(y_train_pred, y_train)
            test_score = r2_score(y_test_pred, y_test)
            
            report[list(models.keys())[i]] = test_score
        return report
    except Exception as e:
        raise CustomeException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomeException(e, sys)