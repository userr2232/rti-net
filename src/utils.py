import pandas as pd
from typing import Optional
import datetime

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