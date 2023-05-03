import pandas as pd
from typing import Optional, Union
from collections.abc import Iterable
import datetime
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path
import joblib
from math import floor

from src.constants import UNIX_EPOCH


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


def get_times(start_date: pd.Timestamp = UNIX_EPOCH(), resolution: int = 1) -> ArrayLike:
    start_time = start_date + pd.Timedelta(19, unit='hours')
    end_time = pd.Timestamp((start_time.date() + pd.Timedelta(1, unit='days'))) + pd.Timedelta(7, unit='hours')
    return pd.date_range(start=start_time, end=end_time, periods=(end_time-start_time).components.hours*(60 // resolution) + 1)[:-1]


def rescale_SNR(df: pd.DataFrame) -> pd.DataFrame:
    df['SNL'] = df['SNL'] * 10.0
    return df


def utc5_offset(df: pd.DataFrame) -> pd.DataFrame:
    UTC5_offset = 5 * 60 * 60
    df['datetime'] = pd.to_datetime(df['UT1_UNIX'] - UTC5_offset, units='s')
    df.drop(columns=['UT1_UNIX'], inplace=True)
    return df


def drop_daytime_obs(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(df.loc[~((df.datetime.dt.hour >= 19) | (df.datetime.dt.hour < 7))].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def drop_heights_out_of_domain(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(df.loc[~((df.GDALT >= 200) & (df.GDALT <= 800))].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def add_day_idx(df: pd.DataFrame) -> pd.DataFrame:
    start_date = UNIX_EPOCH()
    df['day_idx'] = (df.datetime - start_date).dt.components.days
    return df


def read_julia_ESF_data(JULIA_PATH: Union[Path, str], years: Iterable[int]):
    JULIA_PATH = Path(JULIA_PATH)
    file_names = [ f"Julia_ESF_{year}.txt" for year in years ]
    julia_data = [ pd.read_table(JULIA_PATH / file_name, sep='\s+', \
                                na_values='missing', 
                                dtype={'UT1_UNIX': np.int64, \
                                        'GDALT': np.float, \
                                        'SNL': np.float}) \
                    for file_name in file_names ]
    julia_data = [ rescale_SNR(year_data) if year >= 2015 else year_data for year, year_data in zip(years, julia_data) ]
    julia_data = [ utc5_offset(year_data) for year_data in julia_data ]
    julia_data = [ drop_daytime_obs(year_data) for year_data in julia_data ]
    julia_data = [ drop_heights_out_of_domain(year_data) for year_data in julia_data ]
    return [ add_day_idx(year_data) for year_data in julia_data ] 


def get_idx(x: Union[int,float,np.float64,pd.Timestamp], arr: ArrayLike, start_date: pd.Timestamp = UNIX_EPOCH()) -> int:
    if type(x) != float and type(x) != int and type(x) != np.float64:
        if x.hour >= 19:
            x = pd.Timestamp(year=start_date.year, month=start_date.month, day=start_date.day, hour=x.hour, minute=x.minute)
        else:
            x = pd.Timestamp(year=start_date.year, month=start_date.month, day=start_date.day+1, hour=x.hour, minute=x.minute)
    delta = arr[1] - arr[0]
    if x > arr[-1] + delta or x < arr[0]:
        raise ValueError(f"{x} is not within arr's domain [{arr[0]}, {arr[-1]}]")
    n = len(arr)
    lo, hi = 0, n-2
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


def get_idxs(time: pd.Timestamp, height: Union[int,float,np.float64], times: ArrayLike, heights: ArrayLike):
    if not valid_height(height):
        raise ValueError("Height out of bounds")
    if not valid_time(time):
        raise ValueError("Time out of bounds")
    return get_idx(time, times), get_idx(height, heights)


def convert_idx_to_quantity(df: pd.DataFrame, feature_name: str) -> pd.Series:
    if 'time' in feature_name:
        quantity = get_times(resolution=15) # TODO: hardcoded for now
    elif 'height' in feature_name:
        quantity = get_heights(resolution=20) # TODO: hardcoded for now
    else:
        raise ValueError('Feature is not of height or time types.')
    idxs = np.arange(len(quantity))
    idx_to_quantity_mapping = dict(zip(idxs, quantity))
    return df[feature_name].apply(lambda idx: idx_to_quantity_mapping[idx])


def convert_float_idx_to_quantity(df: pd.DataFrame, feature_name: str) -> pd.Series: # this is wrong bc I was assuming that the min start was 0 and the max end was 49 and so on.
    if 'time' in feature_name:
        q_min = UNIX_EPOCH() + pd.Timedelta(19, unit = 'hour')
        q_max = UNIX_EPOCH() + pd.Timedelta(24 + 7, unit = 'hour') + pd.Timedelta(15, unit='minute')
        delta = q_max - q_min
        print("delta", delta)
        quantity = q_min + df[feature_name].apply(lambda f: pd.Timedelta(f * delta.components.hours * 60 * 60 + delta.components.minutes * 60, unit = 'second'))
    elif 'height' in feature_name:
        q_min = 200
        q_max = 820
        delta = q_max - q_min
        quantity = q_min + df[feature_name].apply(lambda f: f * delta)
    else:
        raise ValueError('Feature is not of height or time types.')
    return quantity


def convert_float_idx_to_quantity2(scaled_df: pd.DataFrame, scaler_path: Union[str, Path]) -> pd.DataFrame:
    def add_delta(delta: float, feature_name: str) -> Union[pd.Timedelta, float]:
        if 'time' in feature_name:
            q_delta = pd.Timedelta(round(15 * delta), unit='minute')
        elif 'height' in feature_name:
            q_delta = 20 * delta
        return q_delta
    df = scaled_df.copy()
    scaler = joblib.load(scaler_path)
    print("scaled_df", scaled_df)
    df.loc[:, scaler.columns] = scaler.inverse_transform(scaled_df)
    scaler.columns = [ f'{column}_output' if 'idx' in column else column for column in scaler.columns ]
    df.loc[:, scaler.columns] = scaler.inverse_transform(scaled_df)
    for column in df.columns:
        if 'idx' in column:
            df['tmp'] = df[column] - df[column].apply(lambda x: floor(x))
            df[column] = df[column].apply(lambda x: floor(x))
            df[column] = convert_idx_to_quantity(df = df,
                                                    feature_name = column)
            print("df in loop\n", df)
            df[column] = df.apply(lambda row: row[column] + add_delta(row['tmp'], column), axis=1)
            df.drop('tmp', inplace=True, axis=1)
    df.to_hdf('data.h5', key='data', mode='w') 
    return df
