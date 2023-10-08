import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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
    
    
    
    