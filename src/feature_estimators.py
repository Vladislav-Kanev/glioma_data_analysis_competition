from typing import Literal, Union

import pandas as pd
from yellowbrick.target import FeatureCorrelation


def select_by_correlation_value(correlation_estimator: FeatureCorrelation, min_score: float = 0.5) -> dict[str, float]:
    selected_correlations  = [pair  for pair in zip(correlation_estimator.features_, correlation_estimator.scores_)
                              if abs(pair[1]) > min_score]
    selected_correlations = sorted(selected_correlations, key=lambda x: x[1], reverse=True)
    return dict(selected_correlations)


def get_feature_estimator(training_data: pd.DataFrame, targets: pd.DataFrame,
                          method: Union[Literal['mutual_info-classification'], Literal['pearson']]) -> FeatureCorrelation:
    correlation_estimator = FeatureCorrelation(labels=training_data.columns, method=method, sort=True)
    correlation_estimator.fit(training_data, targets)
    return correlation_estimator


def concat_important_features(important_values_1: Union[list[str], dict[str, float]],
                              important_values_2: Union[list[str], dict[str, float]]) -> list[str]:
    return list(set(important_values_1).union(important_values_2))
