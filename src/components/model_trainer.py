import os
import sys
from dataclasses import dataclass
import logging
import numpy as np

from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from xgboost import XGBClassifier
import lightgbm as lgb
from src.exception import CustomeException

#under sampling

#Grid SearchCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# Scoring
from sklearn.metrics import make_scorer, fbeta_score, recall_score, classification_report, precision_score, confusion_matrix

from src.utils import save_object, tpr_at_fixed_fpr, eval_models, find_best_param, find_best_threshold_for_max_recall_at_fpr



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    calibrated_model_file_path = os.path.join("artifacts","calibrated_model.pkl")
    threshold_file_path = os.path.join("artifacts","threshold.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def init_model_trainer(self, X_train_, X_val_, X_test_, y_train_, y_val_, y_test_):
        try:
            
            X_train = X_train_
            X_val = X_val_
            X_test = X_test_
            y_train = y_train_
            y_val = y_val_
            y_test = y_test_
            
            # models = {
            #     "lightgbm": lgb.LGBMClassifier(),
            #     "xgboost": XGBClassifier(),
            #     "random_forest": RandomForestClassifier(),
                
            # }

            # recall_scorer = make_scorer(recall_score, greater_is_better=True)
            
            # logging.info(f"Starting model evaluation...")
            
            # model_report:dict = eval_models(X_train = X_train, y_train=y_train, 
            #                                 X_test= X_test, y_test= y_test, models= models,
            #                                 custom_scorer = recall_scorer) #change scorer here
            # logging.info(f"Model evaluation complete")
            
            # best_model_score = max(sorted(model_report.values()))
            
            # ## Get best model name
            # best_model_name = list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]
            # best_model = models[best_model_name]
            
            # logging.info(f"Best model: {best_model_name}")
            
            logging.info(f"Finding best parameter for model")
            best_params, best_score = find_best_param(X_train, y_train, X_val, y_val, n_trials=30, n_splits=3, random_state=42)
            logging.info(f"Best parameter found: {best_params} with validation score: {best_score}")
            
            
            # -- Fiting model --
            logging.info(f"Begin fitting model and perform calibration")
            final_model = lgb.LGBMClassifier()
            final_model.set_params(**best_params)
            final_model.fit(X_train, y_train)
            
            # -- Calibrating model on validation set -- 
            calibrated_clf = CalibratedClassifierCV(
                estimator=final_model, cv="prefit", method="sigmoid"
            )
            calibrated_clf.fit(X_val, y_val)
            logging.info(f"Calibration completed")
            
            val_probs = calibrated_clf.predict_proba(X_val)[:, 1]
            
            # Find Maximum recall score at desired FPR
            target_fpr = 0.2 # A fix business requirement threshold, can be change by the higher up
            logging.info(f"Finding maximum recall score at target fpr <= {target_fpr}")
            threshold, fpr, recall_ = find_best_threshold_for_max_recall_at_fpr(
                val_probs, 
                y_val, 
                target_fpr=target_fpr
            )
            
            logging.info(f"Chosen decision threshold (FPR <= {target_fpr}) = {threshold:.3f}")
            logging.info(f"Resulting FPR on validation set: {fpr:.3f}")
            logging.info(f"Resulting Recall on validation set: {recall_:.3f}")
            
            # -- Evaluate on Test set -- 
            test_probs = calibrated_clf.predict_proba(X_test)[:, 1]
            test_preds = (test_probs > threshold).astype(int)
            
            # Precision / Recall
            prec = precision_score(y_test, test_preds)
            rec = recall_score(y_test, test_preds)            
            
            # FPR on Test
            tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, test_preds, labels=[0,1]).ravel()
            denom_test = (tn_test + fp_test) if (tn_test + fp_test) else 1e-10
            test_fpr = fp_test / denom_test
            
            logging.info(f"Test Precision @ threshold={threshold:.3f}: {prec:.3f}")
            logging.info(f"Test Recall    @ threshold={threshold:.3f}: {rec:.3f}")
            logging.info(f"Test FPR       @ threshold={threshold:.3f}: {test_fpr:.3f}")                                       
            
            
            logging.info(f"Saving model, calibrated model and decision threshold to artifacts folder...")
            # Save model, calibrated model, and threshold
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_model
            )
            save_object(
                file_path=self.model_trainer_config.calibrated_model_file_path,
                obj=calibrated_clf
            )
            save_object(
                file_path=self.model_trainer_config.threshold_file_path,
                obj=threshold
            )
            logging.info(f"Saving completed.")
            return calibrated_clf, threshold
        except Exception as e:
            raise CustomeException(e, sys)