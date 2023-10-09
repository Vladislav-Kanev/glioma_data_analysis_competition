import re
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.encoders import encode_dataset


def age_to_float(age: str) -> float:
    pattern = r'(?P<years>\d+) years (?P<days>\d+) days'
    pattern_group = ('years', 'days')

    compiled_pattern = re.compile(pattern)

    if match := compiled_pattern.match(age):
        years = int(match.group(pattern_group[0]))
        days = int(match.group(pattern_group[1]))
        return years + days / 365

    return np.nan


def convert_age(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = [age_to_float(j) for j in dataset]
    return dataset


def normalize_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset[:] = StandardScaler().fit_transform(dataset)
    return dataset.astype(dtype=float)


def get_fill_value(fill_na: str, subset: pd.DataFrame):
    if fill_na == 'median':
        return subset.median()
    elif fill_na == 'mean':
        return subset.mean()
    elif fill_na == 'zero':
        return 0
    raise ValueError('Specified N/A replace is incorrect')


def process_data(dataset: pd.DataFrame,
                 encoder: Literal['OneHot', 'Label'] = 'Label',
                 fill_na: Literal['mean', 'median', 'zero'] = 'median',
                 target: Optional[str] = None
                 ) -> pd.DataFrame:
    dataset = dataset.copy().drop(['Case_ID', 'Primary_Diagnosis'], axis=1)

    dataset['Age_at_diagnosis'] = convert_age(dataset['Age_at_diagnosis'])

    dataset = encode_dataset(dataset, encoder=encoder, target=target)

    for column in dataset.columns:
        fill_value = get_fill_value(fill_na, dataset[column])
        dataset[column] = dataset[column].fillna(fill_value)
    return dataset
