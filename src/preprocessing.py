import pandas as pd
import numpy as np
from typing import Tuple, List, Union, Optional
from pathlib import Path


def UTC_offset(df: pd.DataFrame, offset: float=-5, inplace: bool=False) -> Optional[pd.DataFrame]:
    new_df = df if inplace else df.copy()
    offset *= 60 * 60
    new_df['datetime'] = pd.to_datetime(new_df['UT1_UNIX'] + int(offset), unit='s')
    new_df.drop(columns=['UT1_UNIX'], inplace=True)
    return None if inplace else new_df


def drop_times_out_of_domain(df: pd.DataFrame, domain_start: int=19, domain_end: int=7, inplace: bool=False) -> Optional[pd.DataFrame]:
    new_df = df if inplace else df.copy()
    new_df.drop(new_df.loc[~((new_df.datetime.dt.hour >= domain_start) | (new_df.datetime.dt.hour < domain_end))].index, inplace=True)
    new_df.reset_index(inplace=True, drop=True)
    return None if inplace else new_df


def drop_heights_out_of_domain(df: pd.DataFrame, domain_start: int=200, domain_end: int=800, inplace: bool=False) -> Optional[pd.DataFrame]:
    new_df = df if inplace else df.copy()
    new_df.drop(new_df.loc[~((new_df.GDALT >= domain_start) & (new_df.GDALT <= domain_end))].index, inplace=True)
    new_df.reset_index(inplace=True, drop=True)
    return None if inplace else new_df


def rescale_SNR(df: pd.DataFrame, inplace: bool=False) -> Optional[pd.DataFrame]:
    new_df = df if inplace else df.copy()
    new_df['SNL'] = df['SNL'] * 10.0
    return None if inplace else new_df


def remove_data_from_other_years(df: pd.DataFrame, year: int, inplace: bool=False) -> Optional[pd.DataFrame]:
    new_df = df if inplace else df.copy()
    new_df.drop(new_df.loc[(new_df.datetime.dt.year != year)].index, inplace=True)
    new_df.reset_index(inplace=True, drop=True)
    return None if inplace else new_df


def sort_by_datetime(df: pd.DataFrame, inplace: bool=False) -> Optional[pd.DataFrame]:
    new_df = df if inplace else df.copy()
    new_df.sort_values(by='datetime', inplace=True)
    return None if inplace else new_df


def load(path: Union[Path, str], years: List[int]) -> List[pd.DataFrame]:
    path = Path(path)
    julia_data = [ pd.read_table(path / f"JULIA_ESF_{year}.txt", 
                                    sep='\s+',
                                    na_values='missing', 
                                    dtype={'UT1_UNIX': np.int64, 'GDALT': np.float, 'SNL': np.float}) \
                    for year in years ]
    julia_data = [ UTC_offset(data) for data in julia_data ]
    julia_data = [ remove_data_from_other_years(df=data, year=year) for year, data in zip(years, julia_data) ]
    julia_data = [ rescale_SNR(data) if year >= 2015 else data for year, data in zip(years, julia_data) ]
    julia_data = [ drop_times_out_of_domain(data) for data in julia_data ]
    julia_data = [ drop_heights_out_of_domain(data) for data in julia_data ]
    return julia_data
