import re
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


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


def encode_dataset(dataset: pd.DataFrame, columns: Optional[list[str]] = None,
                   save_state: bool = False, verbose: bool = False
                   ) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict[str, LabelEncoder]]]:

    if not columns:
        columns = list(dataset.columns)

    saved_state: dict[str, LabelEncoder] = {}

    for column in columns:
        encoder = LabelEncoder().fit(dataset[column])
        dataset[column] = encoder.transform(dataset[column])
        if verbose:
            print(f'{column}: {encoder.classes_}')
        if save_state:
            saved_state[column] = encoder

    if save_state:
        return dataset, saved_state
    return dataset



def encode_dataset(dataset: pd.DataFrame, columns: Optional[list[str]] = None,
                   save_state: bool = False, verbose: bool = False
                   ) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict[str, LabelEncoder]]]:

    if not columns:
        columns = list(dataset.columns)

    saved_state: dict[str, LabelEncoder] = {}

    for column in columns:
        encoder = LabelEncoder().fit(dataset[column])
        dataset[column] = encoder.transform(dataset[column])
        if verbose:
            print(f'{column}: {encoder.classes_}')
        if save_state:
            saved_state[column] = encoder

    if save_state:
        return dataset, saved_state
    return dataset


def normalize_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    return StandardScaler().fit_transform(dataset)
