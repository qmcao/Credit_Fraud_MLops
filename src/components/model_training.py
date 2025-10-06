import os
import json
import sys
from dataclasses import dataclass, field
import logging
import numpy as np
import yaml
import pandas as pd
from collections import defaultdict
from itertools import product


from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression


from xgboost import XGBClassifier
import lightgbm as lgb
from src.utils.exception import CustomeException

#under sampling

#Grid SearchCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# Scoring
from sklearn.metrics import make_scorer, recall_score, classification_report, precision_score, confusion_matrix, get_scorer
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils.utils import save_object, save_json

from imblearn.over_sampling import SMOTE


class ModelTrainer:
    def __init__(self):
        '''
        Init model lists and its hyperparameter
        '''
        self.config = self.load_config()
        self.models = [models['name']for models in self.config['model'] ]
        self.params = [models['params'] for models in self.config['model']]
        self.metrics = [metric for metric in self.config['metric']]
        self.primary_metric = self.config['global']['primary_metric']
        self.random_state = self.config['global']['random_state']
        self.n_splits = self.config['global']['n_splits']
        
        # output path
        self.model_out_path = self.config['output']['model']
        self.metrics_out_path = self.config['output']['metrics']
    
    # --steps--: 
    # Split X/y -> model selection via CV (train_data only) -> hyper param tuning 
    # -> refit on full training -> evaluate on testing -> save artifact
    
    def split_feature_target(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        '''
        This function split train_data, test_data into: X_train, X_test, y_train, y_test as numpy array
        '''
        X_train = train_data.drop(['fraud_bool'], axis=1)
        y_train = train_data['fraud_bool']
        X_test = test_data.drop(['fraud_bool'], axis=1)
        y_test = test_data['fraud_bool']
        
        return (np.array(X_train), 
                np.array(X_test), 
                np.array(y_train), 
                np.array(y_test))
        
    
    
    def apply_smote(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        This function apply SMOTE technique to address imbalance problem in response class
        """
        #Init smote obj 
        smote = SMOTE()
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        return X_train_smote, y_train_smote 
    
    
    def train_model(self, X_train: np.ndarray,  y_train: np.ndarray, model_name:str, param_grid: dict):
        """
        Goal: train given model with hyper params tuning, 
        
        return the hyper params with highest cv_score, cv_score, and the fitted model
        
        @ Return:
            grid search best estimator, gs.result
        """
        #store result
        results = []
        
        # cv set up
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Calculate CV scores for each params
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combo in product(*values):
            params = dict(zip(keys, combo))
            print(params)
            fold_scores = defaultdict(list) # automatically creates missing keys with a default empty list.
            
            # Split train_data into train and validation

            # for train_idx, val_idx in cv.split(X_train, y_train):
            for j, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
                print(f"    Fold {j}/{cv.get_n_splits()} ...")
                # print(train_idx)
                # print(X_train[train_idx])
                # return
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                # Apply smote on training data
                X_tr, y_tr = self.apply_smote(X_tr, y_tr)
                
                #init, fit model and test on validate set
                model = self.init_model(model_name, params)
                model.fit(X_tr, y_tr)
                y_pred_prob, y_pred = self.get_scores_and_preds(model, X_val, threshold=0.5)
                
                if "accuracy" in self.metrics:
                    fold_scores["accuracy"].append(accuracy_score(y_val, y_pred))
                if "recall" in self.metrics:
                    fold_scores["recall"].append(recall_score(y_val, y_pred, zero_division=0))
                if "roc-auc" in self.metrics:
                    fold_scores["roc-auc"].append(roc_auc_score(y_val, y_pred_prob))
                
            # mean CV scores for this param set
            mean_scores = {m: float(np.mean(vals)) for m, vals in fold_scores.items()}
            
            results.append({
                "params": params,
                "cv_scores": mean_scores
            })
        
        # return best params and cv_score, then fit the model on the whole training set (with SMOTE)
        _, best_params, best_cv_scores = self.select_best_result(results, self.primary_metric)
        X_train_smote, y_train_smote = self.apply_smote(X_train, y_train)
        model = self.init_model(model_name, best_params)
        model.fit(X_train_smote, y_train_smote)
        
        
        return model, best_params, best_cv_scores, results
    
    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluate model with best hyper params set on test set.
        """
        scores = defaultdict(list) # automatically creates missing keys with a default empty list.
        
        # Predict on X_test and validate with y_test
        y_pred_prob, y_pred = self.get_scores_and_preds(model, X_test)
        if "accuracy" in self.metrics:
            scores["accuracy"].append(accuracy_score(y_test, y_pred))
        if "recall" in self.metrics:
            scores["recall"].append(recall_score(y_test, y_pred))
        if "roc-auc" in self.metrics:
            scores["roc-auc"].append(roc_auc_score(y_test, y_pred_prob))
            
        return scores
    
    def main(self, train_path, test_path):
        '''
        This function orchastrate all the steps in the model_trainer process
        
        This function return best model base on primary metric, 
        its test_score and save the best_model with its metric in the artifacts file
        '''
        
        # read train, test data from artifacts folder and prep training and testing data
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        X_train, X_test, y_train, y_test = self.split_feature_target(train, test)
        
        res = []
        
        for i, model_name in enumerate(self.models):
            model_metrics = {}
            param_grid = self.params[i]
            
            model, best_params, best_cv_scores, results = self.train_model(X_train, y_train, model_name,
                                                                           param_grid)
            model_test_score = self.evaluate(model, X_test, y_test)
            
            # Save metrics of current model
            model_metrics['model'] = model
            model_metrics['model_name'] = model_name
            model_metrics['best_params'] = best_params
            model_metrics['test_score'] = model_test_score
            
            # push the current model metrics to results
            res.append(model_metrics)
        
        # Get best model results
        best_model_index = self.best_test_score_model(res)
        best_model_metric = res[best_model_index]
        
        # Save model and metrics into artifacts folder
        model_fp = os.path.join(self.model_out_path, 'model.pkl')
        metrics_fp = os.path.join(self.metrics_out_path, 'metrics.json')
        save_object(model_fp, best_model_metric['model'])  
        save_json(metrics_fp, best_model_metric['test_score'])
        
        return (model_fp, metrics_fp)
        
    
    
    # -- HELPER -- 
    
    def best_test_score_model(self, results):
        """
        Return index of the model with the best test score
        """
        best_test_score = None
        best_index = None
        
        for i in range(len(results)):
            model_metrics_test_score = results[i]['test_score']
            
            if best_test_score == None or model_metrics_test_score[self.primary_metric] < best_test_score:
                best_test_score = model_metrics_test_score[self.primary_metric]
                best_index = i
        return best_index
        
    
    
    def load_config(self):
        with open('conf/config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def init_model(self, model_name:str, params: dict):
         
        if model_name == "logistic_regression":
            base = LogisticRegression(verbose=0, n_jobs=-1, random_state=self.random_state)
            base.set_params(**params)
            return base
        elif model_name == "random_forest":
            
            base = RandomForestClassifier(random_state=self.random_state)
            base.set_params(**params)
            return base
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        # FUTURE DEV FOR MORE MODEL
        
    
    def get_scores_and_preds(self, model, X: np.ndarray, threshold = 0.5):
        """
        Returns (scores, y_pred).
        - scores: probabilities for positive class
        - y_pred: hard labels thresholded at `threshold`
        """       
        
        if hasattr(model, "predict_proba"):
           y_pred_prob = model.predict_proba(X)[:, 1] # get the predicted prob for positive class only
           
        else:   
            raise NotImplementedError(
                f"{model.__class__.__name__} does not support predict_proba(). "
                "Please use a classifier that supports probability outputs."
            )
        
        y_pred = (y_pred_prob >= threshold).astype(int)
        return y_pred_prob, y_pred

    
    
    def select_best_result(self, all_results, primary_metric="roc-auc"):
        """
        Selects the best param set from CV results based on a primary metric.

        Args:
            all_results (list of dict): each dict has {"params": ..., "cv_scores": {...}}
            primary_metric (str): metric name to optimize

        Returns:
            best (dict): {"params": ..., "cv_scores": {...}}
            best_params (dict)
            best_cv_scores (dict)
        """
        if not all_results:
            raise RuntimeError("No CV results computed.")

        def _score_key(res):
            # if primary_metric missing, treat score as -inf
            return res["cv_scores"].get(primary_metric, -np.inf)

        best = max(all_results, key=_score_key)
        return best, best["params"], best["cv_scores"]
    