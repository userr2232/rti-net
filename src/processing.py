from sqlite3 import Timestamp
import numpy as np
from numpy.typing import ArrayLike
from typing import Union
import pandas as pd


def get_heights(min_height: int = 200, max_height: int = 800, resolution: int = 1) -> ArrayLike:
    return np.linspace(min_height, max_height, (max_height-min_height) // 100 * (100 // resolution) + 1, dtype=np.int16)


def get_times(start_date: pd.Timestamp = pd.Timestamp('1970-01-01'), resolution: int = 1) -> ArrayLike:
    start_time = start_date + pd.Timedelta(19, unit='hours')
    end_time = pd.Timestamp((start_time.date() + pd.Timedelta(1, unit='days'))) + pd.Timedelta(7, unit='hours')
    times = np.linspace(start_time.value, end_time.value, (end_time-start_time).components.hours*(60 // resolution) + 1, dtype=np.int64)
    return pd.to_datetime(times)


def get_idx(x: Union[int,float,np.float64,pd.Timestamp], arr: ArrayLike, start_date: pd.Timestamp = pd.Timestamp('1970-01-01')) -> int:
    if type(x) != float and type(x) != int and type(x) != np.float64:
        if x.hour >= 19:
            x = pd.Timestamp(year=start_date.year, month=start_date.month, day=start_date.day, hour=x.hour, minute=x.minute)
        else:
            x = pd.Timestamp(year=start_date.year, month=start_date.month, day=start_date.day+1, hour=x.hour, minute=x.minute)
    if x > arr[-1] or x < arr[0]:
        raise ValueError("x is not within arr's domain")
    n = len(arr)
    lo, hi = 0, n-2
    delta = arr[1] - arr[0]
    m = None
    while lo <= hi:
        m = (lo + hi) // 2
        if x >= arr[m] and x < arr[m] + delta:
            return m
        if x >= arr[m] + delta:
            lo = m + 1
        else:
            hi = m - 1
    return m