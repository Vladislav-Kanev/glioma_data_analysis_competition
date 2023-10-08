import shutil
from pathlib import Path

import pandas as pd


def get_csv_name(directory: Path, model_name: str, score: float) -> Path:
    model_name = model_name.replace(' ', '_').lower()
    score = str(round(score, 4)).replace('.','_')
    return directory / f'model_{model_name}_{score}.csv'


def convert_result(predictions: list[int], inverse: bool = False) -> pd.DataFrame:
    if inverse:
        predictions = [int(not value) for value in predictions]

    result = pd.DataFrame(zip(range(len(predictions)), predictions), columns=['Id', 'Grade'])
    return result


def new_experiments_folder(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True)
