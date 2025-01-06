import os
import sys
from dataclasses import dataclass
import logging

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

# Scoring
from sklearn.metrics import make_scorer, fbeta_score, recall_score, classification_report

from src.utils import save_object, tpr_at_fixed_fpr, eval_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def init_model_trainer(self, X_train_, X_test_, y_train_, y_test_):
        try:
            X_train = X_train_
            X_test = X_test_
            y_train = y_train_
            y_test = y_test_
            
            models = {
                "lightgbm": lgb.LGBMClassifier(),
                "xgboost": XGBClassifier(),
                "random_forest": RandomForestClassifier(),
                
            }
            
            # If we want to focus on TPR @ 5%FPR, use this custome scorer
            # custom_scorer = make_scorer(
            # tpr_at_fixed_fpr,
            # needs_proba=True,      # We'll receive predicted probabilities
            # greater_is_better=True # We want to maximize TPR
            # )
            recall_scorer = make_scorer(recall_score, greater_is_better=True)
            
            logging.info(f"Starting model evaluation...")
            
            model_report:dict = eval_models(X_train = X_train, y_train=y_train, 
                                            X_test= X_test, y_test= y_test, models= models,
                                            custom_scorer = recall_scorer) #change scorer here
            logging.info(f"Model evaluation complete")
            
            best_model_score = max(sorted(model_report.values()))
            
            ## Get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            logging.info(f"Best model: {best_model_name}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted= best_model.predict(X_test)
            return best_model_name, classification_report(y_test, predicted)
        except Exception as e:
            raise CustomeException(e, sys)