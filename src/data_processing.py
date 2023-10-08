import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def age_to_float(age: str) -> float:
    pattern = r'(?P<years>\d+) years (?P<days>\d+) days'
    pattern_group = ('years', 'days')

    compiled_pattern = re.compile(pattern)

    years = 0
    days = 0

    if match := compiled_pattern.match(age):
        years = int(match.group(pattern_group[0]))
        days = int(match.group(pattern_group[1]))
        
    if not match:
        return np.nan
    
    return years + days / 365


def convert_age(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = [age_to_float(j) for j in dataset]
    return dataset


def normalize_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset[:] = StandardScaler().fit_transform(dataset)
    return dataset.astype(dtype=float)
