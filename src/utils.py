import pandas as pd
from typing import Optional
import datetime
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike


def valid_time(time):
    return time.hour >= 19 or time.hour < 7


def valid_height(height):
    return 200 <= height <= 800


def filter_times(df: pd.DataFrame, date: datetime.date, inplace: bool = False) -> Optional[pd.DataFrame]:
    next_date = date + datetime.timedelta(days=1)
    return df.drop(df.loc[~(((df.datetime.dt.hour >= 19) & (df.datetime.dt.date == date)) 
                            | ((df.datetime.dt.hour < 7) & (df.datetime.dt.date == next_date)))].index, inplace=inplace)


def filter_heights(df: pd.DataFrame, inplace: bool = False) -> Optional[pd.DataFrame]:
    return df.drop(df.loc[~((df.GDALT >= 200)&(df.GDALT <= 800))].index, inplace=inplace)


def filter_times_and_heights(df: pd.DataFrame, date: datetime.date, inplace: bool = False) -> Optional[pd.DataFrame]:
    new_df = df if inplace else df.copy()
    if inplace:
        filter_times(df=new_df, date=date, inplace=True)
        filter_heights(df=new_df, inplace=True)
        return None
    return filter_times(filter_heights(df=new_df), date=date)


class Month(Enum):
    MARCH = 3
    JUNE = 6
    SEPTEMBER = 9
    DECEMBER = 12

    def describe(self):
        return (self.name + " " + ("equinox" if self.value in [3, 9] else "solstice")).title()

    def __str__(self):
        return f"{self.value:02d}"



def get_heights(min_height: int = 200, max_height: int = 800, resolution: int = 1) -> ArrayLike:
    return np.linspace(min_height, max_height, (max_height-min_height) // 100 * (100 // resolution) + 1, dtype=np.int16)[:-1]


def get_times(start_date: pd.Timestamp = pd.Timestamp('1970-01-01'), resolution: int = 1) -> ArrayLike:
    start_time = start_date + pd.Timedelta(19, unit='hours')
    end_time = pd.Timestamp((start_time.date() + pd.Timedelta(1, unit='days'))) + pd.Timedelta(7, unit='hours')
    return pd.date_range(start=start_time, end=end_time, periods=(end_time-start_time).components.hours*(60 // resolution) + 1)[:-1]