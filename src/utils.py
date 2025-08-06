import os
import sys

from imblearn.over_sampling import SMOTE
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

# Calibration library
from sklearn.calibration import CalibratedClassifierCV

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from sklearn.metrics import make_scorer, fbeta_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score
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
    
def run_experiment(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    n_bootstraps=5,
    oversampling=False,
    random_seed=42,
    target_fpr=0.2  # Desired FPR for validation
):
    """
    Runs a training+evaluation experiment across multiple bootstraps, fixing
    a threshold that yields <= target FPR on the validation set, and then
    measuring precision, recall, and FPR on the test set at that threshold.
    
    Parameters:
    -----------
    X_train, y_train : Training features and labels (DataFrame or Series)
    X_val,   y_val   : Validation features and labels
    X_test,  y_test  : Test features and labels
    n_bootstraps     : Number of bootstrap iterations
    oversampling     : Whether to apply SMOTE oversampling in each bootstrap
    random_seed      : Random seed for reproducibility
    target_fpr       : Desired FPR (false positive rate) on the validation set
    
    Returns:
    --------
    precision_list, recall_list, fpr_list : Lists of precision, recall, and FPR across bootstraps
    model : model with best hyper parameter tunning
    """
    
    precision_list = []
    recall_list = []
    fpr_list = []
    decision_threshold = None  # We'll set this only in the first bootstrap
    
    np.random.seed(random_seed)  # Reproducibility
    
    best_params = {} # We'll set this only in the first bootstrap
    
    for i in range(n_bootstraps):
        print(f"\n=== Bootstrap Iteration #{i} (Oversampling={oversampling}) ===")
        
        # ---------------------------
        # 1) Bootstrapping
        # ---------------------------
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_train_boot = X_train.iloc[indices]
        y_train_boot = y_train.iloc[indices]

        
        # 
        #  ---------------------------
        # 2) (Optional) Oversampling
        # ---------------------------
        if oversampling:
            smote_obj = SMOTE(random_state=i)  # different random state each iteration
            X_train_boot, y_train_boot = smote_obj.fit_resample(X_train_boot, y_train_boot)
            
        # ---------------------------
        # 3) Hyper Parameter Tuning for first boostraps using validation set (month 6)
        # ---------------------------        
        if i == 0:
            def objective(trial):
                params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
                    'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                }
                
                # --- Classifier ---
                optuna_model = lgb(random_state = i)
                optuna_model.set_params(**params)
                #fit model
                optuna_model.fit(
                    X=X_train_boot,
                    y=y_train_boot,
                )
                
                # Predict on the validation set
                y_pred_proba = optuna_model.predict_proba(X_val)[:, 1]
                
                # Return the validation ROC_AUC score as the optimization objective
                score = roc_auc_score(y_val, y_pred_proba)
                
                
                return score.mean()

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=30)
                    
            best_params = study.best_params
        
        # ---------------------------
        # 4) Model Training with best param from first iterations
        # ---------------------------
        
        #print(best_params)
        tuned_lgb = lgb(random_state=i)
        tuned_lgb.set_params(**best_params)
        tuned_lgb.fit(X_train_boot, y_train_boot)     
        
        # ---------------------------
        # 5) Calibration (Platt / sigmoid) on the 1st bootstrap with validation set set to define threshold. 
        # We then reapply the same threshold for the remaining bootstraps
        # ---------------------------
        calibrated_clf = CalibratedClassifierCV(
            estimator=tuned_lgb, cv="prefit", method="sigmoid"
        )
        calibrated_clf.fit(X_val, y_val)        
        
        if i == 0:
            # calibrated_clf = CalibratedClassifierCV(
            #     base_estimator=base_lgb, cv='prefit', method='sigmoid') #this code is wrong, base_estimator is no longer support!, takes 3 hours to debug
            
            val_probs = calibrated_clf.predict_proba(X_val)[:, 1]
            
            # Maximum recall score at desired FPR
            threshold, fpr, recall_ = find_best_threshold_for_max_recall_at_fpr(
                val_probs, 
                y_val, 
                target_fpr=target_fpr
            )
            
            decision_threshold = threshold
            print(f"Chosen decision threshold (FPR <= {target_fpr}) = {decision_threshold:.3f}")
            print(f"Resulting FPR on validation set: {fpr:.3f}")
            print(f"Resulting Recall on validation set: {recall_:.3f}")
        
        # ---------------------------
        # 6) Evaluate on Test set for Model #k > 0
        # ---------------------------       
        else:
            test_probs = calibrated_clf.predict_proba(X_test)[:, 1]
            test_preds = (test_probs > decision_threshold).astype(int)
            
            # Precision / Recall
            prec = precision_score(y_test, test_preds)
            rec = recall_score(y_test, test_preds)
            
            # FPR on Test
            tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, test_preds, labels=[0,1]).ravel()
            denom_test = (tn_test + fp_test) if (tn_test + fp_test) else 1e-10
            test_fpr = fp_test / denom_test
            
            precision_list.append(prec)
            recall_list.append(rec)
            fpr_list.append(test_fpr)
            
            print(f"Test Precision @ threshold={decision_threshold:.3f}: {prec:.3f}")
            print(f"Test Recall    @ threshold={decision_threshold:.3f}: {rec:.3f}")
            print(f"Test FPR       @ threshold={decision_threshold:.3f}: {test_fpr:.3f}")
    
    return precision_list, recall_list, fpr_list

def find_best_threshold_for_max_recall_at_fpr(
    val_probs, y_val, 
    target_fpr=0.05
):
    """
    Given validation set probabilities (val_probs) and ground truth (y_val),
    find the threshold that yields the highest recall subject to FPR <= target_fpr.
    
    Parameters
    ----------
    val_probs : np.ndarray
        Predicted probabilities for the positive class on the validation set.
    y_val : np.ndarray
        True labels (0 or 1) for the validation set.
    target_fpr : float
        The maximum allowed false positive rate.
        
    Returns
    -------
    best_threshold : float
        The threshold that yields the maximum recall while keeping FPR <= target_fpr.
    best_fpr : float
        The FPR at that threshold.
    best_recall : float
        The recall at that threshold.
    """
    
    try:
        # Sort the probabilities in ascending order
        sorted_thresholds = np.sort(val_probs)
        
        best_threshold = 0.0
        best_fpr = 1.0
        best_recall = 0.0
        
        # We'll try each threshold in ascending order.
        # For each threshold, we measure FPR and recall.
        for t in sorted_thresholds:
            preds = (val_probs >= t).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0,1]).ravel()
            
            # Compute FPR = FP / (FP + TN)
            # Avoid division by zero if TN+FP=0
            denom = (fp + tn) if (fp+tn) else 1e-15
            current_fpr = fp / denom
            
            # Compute recall = TP / (TP + FN)
            # Avoid division by zero if TP+FN=0
            denom_pos = (tp + fn) if (tp+fn) else 1e-15
            current_recall = tp / denom_pos
            
            # We only consider thresholds where FPR <= target_fpr
            if current_fpr <= target_fpr:
                # Among those, pick the one that yields the highest recall
                if current_recall > best_recall:
                    best_threshold = t
                    best_fpr = current_fpr
                    best_recall = current_recall
        
        return best_threshold, best_fpr, best_recall

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


def find_best_param(X_train, y_train, X_val, y_val, n_trials = 20, n_splits = 3, random_state=42):
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
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
                'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            }
            
            # --- Classifier ---
            optuna_model = lgb.LGBMClassifier(random_state = random_state)
            optuna_model.set_params(**params)
            #fit model
            optuna_model.fit(
                X=X_train,
                y=y_train,
            )
            
            # Predict on the validation set
            y_pred_proba = optuna_model.predict_proba(X_val)[:, 1]                   
            # Return the validation ROC_AUC score as the optimization objective
            score = roc_auc_score(y_val, y_pred_proba)             
            return score
        except Exception as e:
            raise CustomeException(e, sys)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
            
    best_params = study.best_params  
    best_score = study.best_trials              
    
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