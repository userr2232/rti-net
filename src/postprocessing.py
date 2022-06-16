from typing import Union
from pathlib import Path
import re
import os
import pandas as pd
import numpy as np
from src.utils import get_times


def daily_npys_to_1D_occurrence_df(path: Union[Path, str]) -> pd.DataFrame:
    pattern = r"^\d\d\d\d-\d\d-\d\d.npy$"
    re_obj = re.compile(pattern)
    _, _, filenames = next(os.walk(path), (None, None, []))
    df = pd.DataFrame({})
    for filename in filenames:
        if re_obj.fullmatch(filename):
            rti = np.load(path / filename)
            if rti.sum(axis=1).shape[0] == 48:
                day_df = pd.DataFrame({'ESF': rti.sum(axis=1), 
                                        'LT': get_times(pd.Timestamp(filename[:-4]), resolution=15)})
            else:
                day_df = pd.DataFrame({'ESF': rti.sum(axis=1)[:-1], 
                                        'LT': get_times(pd.Timestamp(filename[:-4]), resolution=15)})
            df = pd.concat([df, day_df], ignore_index=True)
    return df