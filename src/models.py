import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def logreg_classifier(train_data:pd.DataFrame, target:pd.DataFrame):
    parameters = {
        'penalty' : ['l2'],
        'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    }
    model = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(model, parameters)
    grid_search.fit(train_data, target)

    return grid_search



def catboost_classifier(dataset: pd.DataFrame, targets: pd.DataFrame, test_size=0.2):
    catboost = CatBoostClassifier()

    param_grid = {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [6, 8, 10],
    }

    grid_search = GridSearchCV(
        catboost, param_grid, cv=5, scoring='f1', n_jobs=-1)

    res = grid_search.fit(dataset, targets)

    return res
