import re

import numpy as np


def get_float_age(age: str) -> float:
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
