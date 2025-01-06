import os
import sys


from catboost import CatBoostClassifier

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import lightgbm as lgb
from xgboost import XGBClassifier

#under sampling
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline as ImbPipeline


import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from sklearn.metrics import make_scorer, fbeta_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
import optuna
from src.exception import CustomeException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomeException(e, sys)
    


def tpr_at_fixed_fpr(y_true, y_prob, fpr_target=0.05):
    """
    Returns the maximum TPR achieved at or below fpr_target (e.g. 0.05).
    
    :param y_true:   True binary labels (0 or 1).
    :param y_proba:  Predicted probabilities of class=1.
    :param fpr_target: The target (maximum) false-positive rate (e.g. 0.05).
    :return: The highest TPR achievable with FPR <= fpr_target.
    """  
    try:
        fpr, tpr, thres = roc_curve(y_true, y_prob)
    
        # We want the maximum TPR such that FPR <= target_FPR
        mask = (fpr <= fpr_target)
    
        if np.any(mask):
            return np.max(tpr[mask]) # Best TPR among those points
        else:
            return 0
    except Exception as e:
        raise CustomeException(e, sys)


def find_best_param(X_train, y_train, model_name, custom_scorer, n_trials = 20, n_splits = 3, random_state=42):
    """
    Finds the best hyperparameters for the specified model using Optuna.

    @Parameters:
    - X_train: Training features.
    - y_train: Training labels.
    - model_type: String specifying the model ('lightgbm', 'xgboost', 'random_forest').
    - custom_scorer: A custom scoring function compatible with scikit-learn.
    - n_trials: Number of Optuna trials.
    - n_splits: Number of cross-validation splits.
    - random_state: Random state for reproducibility.

    Returns:
    - best_params: Dictionary of the best hyperparameters found.
    - best_score: Best score achieved with the best hyperparameters.
    """
    def objective(trial):
        try:
            if model_name == "lightgbm":
                # --- Hyperparam suggestions for LightGBM---
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 7, 20),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10, log=True),
                    'random_state': random_state
                }
                model = lgb.LGBMClassifier(**params)
                
            elif model_name == 'xgboost':
                # --- hyper param for XGboost----
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10, log=True),
                    'random_state': random_state,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }
                model = XGBClassifier(**params)            
        
            elif model_name == 'random_forest':
                # --- Hyperparameter Suggestions for Random Forest ---
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'random_state': random_state
                }
                model = RandomForestClassifier(**params)
            else:
                raise ValueError(f"Unsupported model_type: {model_name}")
            
            # --- Build Pipeline: NearMiss + Classifier ---
            pipeline = ImbPipeline([
                
                ('resampler', NearMiss()),
                ('classifier', model)
            ])

            # --- Cross-Validation ---
            skf = StratifiedKFold(n_splits=n_splits, shuffle= True, random_state=random_state )
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                scoring=custom_scorer,
                cv = skf,
                n_jobs=-1
            )
            # --- Return the Mean Score ---
            return scores.mean()

        except Exception as e:
            raise CustomeException(e, sys)
        
    #init optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials= n_trials)  # Try different hyperparameter sets
    
    #-- Retrieve Best Param and Score
    best_params = study.best_params
    best_score = study.best_value
    
    return best_params, best_score
    
    
    
def eval_models(X_train, y_train, X_test, y_test, models, custom_scorer):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
                
            param, _ = find_best_param(
                X_train, 
                y_train, 
                model_name, 
                custom_scorer = custom_scorer, 
                n_trials = 10, 
                n_splits = 3, 
                random_state=42)
            #param = list(params.values())[i]
            
            #gs = GridSearchCV(model,param,cv=3)
            #gs.fit(X_train, y_train)
            
            model.set_params(**param)
            final_pipeline = ImbPipeline([
                ('resampler', NearMiss()),
                ('classifier', model)
            ])
            
            final_pipeline.fit(X_train, y_train)
            
            #y_train_pred = final_pipeline.predict(X_train)
            y_test_pred = final_pipeline.predict(X_test)
            
            #train_score = recall_score(y_train_pred, y_train)
            test_score = recall_score(y_test, y_test_pred)
            
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