from typing import Literal, Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def encode_dataset(dataset: pd.DataFrame, encoder: Literal['OneHot', 'Label'],
                   target: str = 'Grade') -> pd.DataFrame:
    object_columns = list(dataset.select_dtypes(include='object').columns)

    if encoder == 'OneHot':
        object_columns_copy = object_columns.copy()
        object_columns_copy.remove(target)
        encoded_dataset = onehot_encode_dataset(dataset, object_columns_copy)
        encoded_dataset = label_encode_dataset(encoded_dataset, [target])
    else:
        encoded_dataset = label_encode_dataset(dataset, object_columns)
    return encoded_dataset


def onehot_encode_dataset(dataset: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
    if not columns:
        columns = list(dataset.columns)
    
    to_encoded = dataset[columns].reset_index(drop=True)
    not_encoded = dataset.drop(columns=columns).reset_index(drop=True)

    encoder = OneHotEncoder().fit(to_encoded)
    transformed = encoder.transform(to_encoded)

    encoded = pd.DataFrame(transformed.toarray(), columns=encoder.get_feature_names_out())
    dataset = pd.concat([not_encoded, encoded], axis=1)
    return dataset


def label_encode_dataset(dataset: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
    if not columns:
        columns = list(dataset.columns)

    for column in columns:
        encoder = LabelEncoder().fit(dataset[column])
        dataset[column] = encoder.transform(dataset[column])
    return dataset
