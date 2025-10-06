import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
import json

from src.utils.exception import CustomeException



def save_object(file_path, obj):

    dir_path = os.path.dirname(file_path)

    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)

    return file_path

def save_json(file_path: str, data: dict):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    
    return file_path
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomeException(e, sys)