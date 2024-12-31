class_weight = [{0: 0.2, 1: 0.8},
                {0: 0.4, 1: 0.6},
                {0: 0.7, 1: 0.3}
                ]

#Regularization param
classifier_c = [0.01]


param_grid = [
    {
        "preprocessing__num_pipeline__num_feature_select__k": fixed_k_num,
        "preprocessing__cat_pipeline__cat_feature_select__k": fixed_k_cat,
    },
    
    {
        "classifier__C": classifier_c, # Previously find best C = 0.01
        "classifier__class_weight": class_weight
    }

]


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=final_pipeline_undersampling,
    param_grid=param_grid,
    scoring=f1_scorer, # can try different scoring
    cv=skf,
    n_jobs=-1,
    refit=True
)