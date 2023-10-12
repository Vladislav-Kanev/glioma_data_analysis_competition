from typing import Literal

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def grid_search(estimator_name: Literal['RandomForest', 'CatBoost', 'LogisticRegression'],
                train_data: pd.DataFrame, target: pd.DataFrame) -> BaseEstimator:

    model_parameter_map = {
        'RandomForest': (RandomForestClassifier(), {
            'n_estimators': [50, 100, 200, 300, 400],
            'max_depth': [6, 8, 10]}),
        'CatBoost': (CatBoostClassifier(verbose=False), {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [6, 8, 10]}),
        'LogisticRegression': (LogisticRegression(), {
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}),
    }

    model, parameters = model_parameter_map[estimator_name]

    grid_search = GridSearchCV(
        model, parameters, cv=5, scoring='f1', n_jobs=-1)
    best_estimator = grid_search.fit(train_data, target)

    model = model.set_params(
        **best_estimator.best_params_).fit(train_data, target)

    return model


def voting_search(train_data: pd.DataFrame, target: pd.DataFrame):
    models = [
        ('RandomForest', RandomForestClassifier(n_estimators=100)),
        ('CatBoost', CatBoostClassifier(verbose=False)),
        ('LogisticRegression', LogisticRegression())
    ]

    ensemble_model = VotingClassifier(estimators=models, voting='hard')
    model = ensemble_model.fit(train_data, target)
    return model
