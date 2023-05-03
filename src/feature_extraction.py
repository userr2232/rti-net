from pathlib import Path
from enum import Enum, unique
import h5py
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator, MaxNLocator
from matplotlib.figure import figaspect
import matplotlib.dates as mdates
import matplotlib.dates as mdates
import matplotlib as mpl
from typing import List, Tuple, Union
import os

from src.utils import get_idxs, get_times, get_heights
from src.constants import UNIX_EPOCH



def rti_thresholding(data: pd.DataFrame, times: ArrayLike, heights: ArrayLike) -> Tuple[ArrayLike,pd.DataFrame]:
    rti_count = np.zeros((len(times), len(heights)))
    s = set()
    for _, height, snr, time, _ in data.itertuples():
        s.add(time.date())
        if snr <= -20: continue
        time_idx, height_idx = get_idxs(time=time, height=height, times=times, heights=heights)
        rti_count[time_idx][height_idx] += 1
    rti_map = rti_count > 10
    date = UNIX_EPOCH() + pd.Timedelta(data.day_idx.iloc[0], unit='days')
    deltas = times - times[0]
    timestamps = date + pd.Timedelta(19, unit='hours') + deltas
    height_collapsed_rti = rti_map.sum(axis=1)
    rti_df = pd.DataFrame({'LT': timestamps, 'ESF': height_collapsed_rti})
    return rti_map, rti_df


# def onset_and_end_time_and_height(df: pd.DataFrame, times: ArrayLike, heights: ArrayLike) -> pd.DataFrame:
#     lowres_rti, _ = rti_thresholding(data=df, times=times, heights=heights)
#     if lowres_rti.sum() == 0: return None

#     time_resolution = times[1] - times[0]
#     height_resolution = heights[1] - heights[0]

#     ESF_occurrences = lowres_rti.sum(axis=1) > 0
#     onset_time_idx, *_, end_time_idx = np.arange(len(times))[ESF_occurrences]

#     initial_ESF_heights = lowres_rti[onset_time_idx,:]
#     end_ESF_heights = lowres_rti[end_time_idx,:]

#     date = UNIX_EPOCH() + pd.Timedeta(df.day_idx.iloc[0], unit='days')
#     deltas = times - times[0]
#     timestamps = pd.Timestamp(date) + pd.Timedelta(19, unit='hours') + deltas

#     onset_time_range = (timestamps[ESF_occurrences][0], timestamps[ESF_occurrences][0] + time_resolution)
#     end_time_range = (timestamps[ESF_occurrences][-1], timestamps[ESF_occurrences][-1] + time_resolution)

#     onset_height_range = (heights[initial_ESF_heights][0], heights[initial_ESF_heights][0] + height_resolution)

#     return df.loc[(
#         (df.datetime >= onset_time_range[0]) & (df.datetime < onset_time_range[1]) &
#         (df.GDALT >= onset_height_range[0]) & (df.GDALT < onset_height_range[1])
#     )].sort_values(['datetime', 'GDALT']).iloc[0].to_frame().T, \
#         df.loc[(
#             (df.datetime >= end_time_range[0]) & (df.datetime < end_time_range[1])
#         )].sort_values('datetime', ascending=False).iloc[0].to_frame().T


# def onset_and_end_times_and_heights(julia_data: List[pd.DataFrame]) -> pd.DataFrame:
#     df = pd.DataFrame()
#     times15 = get_times(resolution=15)
#     heights20 = get_heights(resolution=20)
#     for year_data in julia_data:
#         for _, day_data in year_data.groupby(['day_idx']).__iter__():
#             tmp_df = onset_and_end_time_and_height(df=day_data,
#                                             times=times15,
#                                             heights=heights20)
#             if tmp_df is not None:
#                 df = df.append(tmp_df, ignore_index=True)
#     return df


def time_occurrence_idxs(rti_map: ArrayLike, times: ArrayLike) -> Tuple[int]:
    time_occurrence = rti_map.sum(axis=1) > 0
    if time_occurrence.sum() >= 2:
        onset_idx, *_, end_idx = np.arange(len(times))[time_occurrence]
    else:
        onset_idx = end_idx = np.arange(len(times))[time_occurrence][0]
    return onset_idx, end_idx


def height_occurrence_idx(rti_map: ArrayLike, heights: ArrayLike) -> int:
    height_occurrence = rti_map.sum(axis=0) > 0
    *_, plume_idx = np.arange(len(heights))[height_occurrence]
    return plume_idx


def extract_features(julia_data: List[pd.DataFrame]) -> pd.DataFrame:
    df = pd.DataFrame()
    times15, heights20 = get_times(resolution=15), get_heights(resolution=20)
    for year_data in julia_data:
        for day_idx, day_data in year_data.groupby(['day_idx']).__iter__():
            rti_map, _ = rti_thresholding(data=day_data, times=times15, heights=heights20)
            if rti_map.sum() == 0:
                continue
            onset_time_idx, end_time_idx = time_occurrence_idxs(rti_map=rti_map, times=times15)
            max_height_idx = height_occurrence_idx(rti_map=rti_map, heights=heights20)
            new_row = pd.Series({'day_idx': day_idx, 'onset_time_idx': onset_time_idx, 'end_time_idx': end_time_idx, 'max_height_idx': max_height_idx})
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    return df


def extract_features_from_processed_data(path: Union[Path,str]) -> pd.DataFrame:
    path = Path(path)
    df = pd.DataFrame()
    times15, heights20 = get_times(resolution=15), get_heights(resolution=20)
    for filename in os.listdir(path):
        if filename.endswith(".npy") and filename[7] == '-': # could be '-' or '_', see process function
            rti_map = np.load(path / filename)[:-1,:-1] > 0
            if rti_map.sum() == 0:
                continue
            onset_time_idx, end_time_idx = time_occurrence_idxs(rti_map=rti_map, times=times15)
            max_height_idx = height_occurrence_idx(rti_map=rti_map, heights=heights20)
            new_row = pd.Series({'date': filename.split('.')[0], 'onset_time_idx': onset_time_idx, 'end_time_idx': end_time_idx, 'max_height_idx': max_height_idx})
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    df['date'] = pd.to_datetime(df.date, format="%Y-%m-%d")
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
