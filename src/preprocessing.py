import pandas as pd
import numpy as np
from typing import Tuple, List, Union, Optional
from pathlib import Path
from omegaconf import DictConfig
import pyarrow as pa
import re
import h5py
import os
from pyarrow import dataset as ds
from src.utils import add_day_idx
from src.feature_extraction import extract_features_from_processed_data


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
    julia_data = [ pd.read_table(path / f"Julia_ESF_{year}.txt", 
                                    sep='\s+',
                                    na_values='missing', 
                                    dtype={'UT1_UNIX': np.int64, 'GDALT': np.float, 'SNL': np.float}) \
                    for year in years ]
    julia_data = [ UTC_offset(data) for data in julia_data ]
    julia_data = [ remove_data_from_other_years(df=data, year=year) for year, data in zip(years, julia_data) ]
    julia_data = [ rescale_SNR(data) if year >= 2015 else data for year, data in zip(years, julia_data) ]
    julia_data = [ drop_times_out_of_domain(data) for data in julia_data ]
    julia_data = [ drop_heights_out_of_domain(data) for data in julia_data ]
    return [ add_day_idx(year_data) for year_data in julia_data ]


def preprocessing(cfg: DictConfig, save: bool = False, path: Optional[Union[str, Path]] = None) -> pa.Table:
    if save:
        assert(path is not None)
        path = Path(path)
    
    JULIA_PATH = Path(cfg.datasets.julia)
    
    pattern = r"^\d\d_\d\d\d\d.csv$"
    re_obj = re.compile(pattern)
    _, _, filenames = next(os.walk(JULIA_PATH), (None, None, []))
    fabiano_ESF = pd.DataFrame({})
    for filename in filenames:
        if re_obj.fullmatch(filename):
            season_df = pd.read_csv(JULIA_PATH / filename, parse_dates=['LT'], infer_datetime_format=True)
            if not season_df.empty:
                fabiano_ESF = pd.concat([fabiano_ESF, season_df], ignore_index=True)
    fabiano_ESF.LT = pd.to_datetime(fabiano_ESF.LT)
    print(fabiano_ESF)
    rti_features = extract_features_from_processed_data(cfg.datasets.processed)
    print("rti_features dtypes", rti_features.dtypes)
    fabiano_ESF.sort_values('LT', inplace=True)
    fabiano_ESF.reset_index(drop=True, inplace=True)
    h5_d = h5py.File(cfg.datasets.geo_param, 'r')
    df = pd.DataFrame(h5_d['Data']['GEO_param'][()])
    df.loc[:, 'date_hour'] = pd.to_datetime(df.loc[:, ('YEAR', 'MONTH', 'DAY', 'HOUR')])
    df['LT'] = df.date_hour - pd.Timedelta("5h")
    df.drop('date_hour', axis=1, inplace=True)
    df.sort_values('LT', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(['YEAR', 'MONTH', 'DAY', 'HOUR'], axis=1, inplace=True)
    print("DF", df)
    df2 = pd.DataFrame(h5_d['Data']['SAO_total'][()])
    df2.rename(columns={'MIN': 'MINUTE', 'SEC': 'SECOND'}, inplace=True)
    df2.loc[:, 'datetime'] = pd.to_datetime(df2.loc[:,('YEAR','MONTH','DAY','HOUR','MINUTE')])
    df2.sort_values('datetime', inplace=True)
    df2.reset_index(drop=True, inplace=True)
    df2['LT'] = df2.datetime - pd.Timedelta('5h')
    df2.drop('datetime', axis=1, inplace=True)
    df2.drop(['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND', 'foF1',
    'hmF1','foE','hmE','V_hF2','V_hE','V_hEs','hmF2'], axis=1, inplace=True)
    df2.dropna(inplace=True)
    df2.reset_index(drop=True, inplace=True)
    df.drop(['DST', 'KP'], axis=1, inplace=True)
    df2.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("DF 2", df2)
    df3 = pd.merge_asof(df2, df, on='LT', tolerance=pd.Timedelta('59m'))
    print("DF 3 right after merge", df3)
    df3.index = df3.LT
    df3.drop(df3.loc[((df3['F10.7'].isna())|(df3.AP.isna()))].index, inplace=True)
    df3['F10.7 (90d)'] = df3['F10.7'].rolling('90d').mean()
    df3['F10.7 (90d dev.)'] = df3['F10.7'] - df3['F10.7 (90d)']
    df3['AP (24h)'] = df3['AP'].rolling('24h').mean()
    df3['V_hF_prev'] = df3['V_hF'].rolling('30min').agg(lambda rows: rows[0])
    df3['V_hF_prev_time'] = df3['V_hF'].rolling('30min').agg(lambda rows: pd.to_datetime(rows.index[0]).value)
    df3['V_hF_prev_time'] = pd.to_datetime(df3['V_hF_prev_time'])
    df3.reset_index(drop=True, inplace=True)
    df3['delta_hF'] = df3['V_hF']-df3['V_hF_prev']
    df3['delta_time'] = (df3['LT']-df3['V_hF_prev_time']).dt.components.minutes + 1e-9
    df3['delta_hF_div_delta_time'] = df3['delta_hF'] / df3['delta_time']
    df3.drop(['delta_hF', 'delta_time', 'V_hF_prev_time'], axis=1, inplace=True)
    print("DF 3", df3)
    delta_hours = cfg.preprocessing.delta_hours
    fabiano_ESF[f'LT-{delta_hours}h'] = fabiano_ESF.LT - pd.Timedelta(f'{delta_hours}h')
    fabiano_ESF.index = fabiano_ESF.LT
    fabiano_ESF['accum_ESF'] = fabiano_ESF.ESF.rolling('1h').sum()
    fabiano_ESF.reset_index(drop=True, inplace=True)
    merged = pd.merge_asof(fabiano_ESF, df3, 
                            left_on=f'LT-{delta_hours}h', right_on='LT', 
                            tolerance=pd.Timedelta('15m'), 
                            direction='nearest')
    merged.dropna(inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged = merged.loc[((merged.LT_y.dt.hour == 19)&(merged.LT_y.dt.minute == 30))].copy()
    merged['date'] = pd.to_datetime(merged.LT_y.dt.date)
    print("merged dtypes", merged.dtypes)
    FIRST2_0 = pd.merge(merged, rti_features, on='date')
    FIRST2_0.dropna(inplace=True)
    FIRST2_0.reset_index(drop=True, inplace=True)
    FIRST2_0['year'] = FIRST2_0.LT_y.dt.year
    print("FIRST2_0", FIRST2_0)
    
    data = pa.Table.from_pandas(FIRST2_0)
    if save:
        partitioning = ds.partitioning(pa.schema([("year", pa.int16())]), flavor="hive")
        ds.write_dataset(data, str(path / "partitioned"), format="ipc", 
                         partitioning=partitioning, existing_data_behavior="delete_matching")
    return data