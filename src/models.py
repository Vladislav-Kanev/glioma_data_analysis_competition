import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


def logreg_classifier(train_data: pd.DataFrame, target: pd.DataFrame):
    parameters = {
        'penalty' : ['l2'],
        'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    }

    model = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(model, parameters)
    best_estimator = grid_search.fit(train_data, target)

    return best_estimator


def catboost_classifier(dataset: pd.DataFrame, targets: pd.DataFrame):
    parameters = {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [6, 8, 10],
    }

    model = CatBoostClassifier(verbose=False)
    grid_search = GridSearchCV(model, parameters, cv=5, scoring='f1', n_jobs=-1)
    best_estimator = grid_search.fit(dataset, targets)

    return best_estimator


def rf_classifier(train_data: pd.DataFrame, target: pd.DataFrame):
    parameters = {
        'n_estimators': [50, 100, 200, 300, 400],
        'max_depth' : [6, 8, 10],
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, parameters)
    best_estimator = grid_search.fit(train_data, target)

    return best_estimator


def voting_classifier(estimators: list, train_data: pd.DataFrame, target: pd.DataFrame):
    ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
    model = ensemble_model.fit(train_data, target)
    return model