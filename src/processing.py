from operator import attrgetter
from re import L
import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Tuple, Optional, List
import pandas as pd
from src.utils import valid_time, valid_height, filter_times_and_heights
from enum import Enum
from pathlib import Path
from src.preprocessing import load
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import datetime
import logging

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
    return np.linspace(min_height, max_height, (max_height-min_height) // 100 * (100 // resolution) + 1, dtype=np.int16)


def get_times(start_date: pd.Timestamp = pd.Timestamp('1970-01-01'), resolution: int = 1) -> ArrayLike:
    start_time = start_date + pd.Timedelta(19, unit='hours')
    end_time = pd.Timestamp((start_time.date() + pd.Timedelta(1, unit='days'))) + pd.Timedelta(7, unit='hours')
    return pd.date_range(start=start_time, end=end_time, periods=(end_time-start_time).components.hours*(60 // resolution) + 1)


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


def get_idxs(time: pd.Timestamp, height: Union[int,float,np.float64], times: ArrayLike, heights: ArrayLike):
    if not valid_height(height):
        raise ValueError("Height out of bounds")
    if not valid_time(time):
        raise ValueError("Time out of bounds")
    return get_idx(time, times), get_idx(height, heights)


def process_day(data: pd.DataFrame, times: ArrayLike, heights: ArrayLike, save: bool = False, path: Union[Path,str] = None) -> Optional[ArrayLike]:
    if data.empty:
        return None
    if save:
        if path is None:
            raise ValueError("You must include a path if save argument is True.")
        else:
            path = Path(path)
    ESF_count = np.zeros((len(times), len(heights)))
    for row in data.itertuples():
        time, height, snr = attrgetter('datetime', 'GDALT', 'SNL')(row)
        if snr <= -20: continue
        time_idx, height_idx = get_idxs(time=time, height=height, times=times, heights=heights)
        ESF_count[time_idx][height_idx] += 1
    ESF_bitmap = ESF_count > 10
    if save:
        np.save(path, ESF_bitmap)
    return ESF_bitmap


def process_season(year: int, month: Month, data: pd.DataFrame, save: bool = False, path: Union[str,Path] = None) -> Tuple[ArrayLike, ArrayLike, int]:
    logging.info(f'processing season {year}-{month}')
    if save:
        if path is None:
            raise ValueError("You must include a path if save argument is True.")
        else:
            path = Path(path)
    median_date = pd.Timestamp(f"{year}-{month}-21")
    median_delta = pd.Timedelta(45, unit='days')
    season_start, season_end = median_date - median_delta, median_date + median_delta
    season = data.loc[((data.datetime >= season_start) & (data.datetime <= season_end))]
    n_observed_days = 0
    heights, times = get_heights(resolution=20), get_times(resolution=15)
    total_occurrence = np.zeros((len(times), len(heights)))
    dates = pd.date_range(start=season_start, end=season_end, periods=(season_end - season_start).components.days + 1)
    for date in dates:
        days_data = filter_times_and_heights(df=season, date=date.date())
        logging.info(f'processing day {date.date()}')
        days_ESF_bitmap = process_day(data=days_data, times=times, heights=heights, save=save, path=path / (str(date.date())) if path is not None else None)
        logging.info(f'proccesed day {date.date()}')
        if days_ESF_bitmap is not None:
            total_occurrence += days_ESF_bitmap
            n_observed_days += 1
    occurrence_rate = np.empty_like(total_occurrence)
    occurrence_rate[:,:] = np.nan
    if n_observed_days > 0 and not ((month == 12 and year == 2020) or 
                                    (month == 3 and year == 2002) or
                                    (month == 6 and (year == 2000 or year == 2001 or year == 2013))):
        occurrence_rate = total_occurrence / n_observed_days
    if save:
        np.save(path / f"{year}-{month}_{n_observed_days}", total_occurrence)
    logging.info(f'processed season {year}-{month}')
    return total_occurrence, occurrence_rate, n_observed_days


def plot_season(data: pd.DataFrame, year: int, month: Month, save: bool = False, path: Union[str,Path] = None) -> None:
    logging.info(f'plotting season {year}-{month}')
    if save:
        if path is None:
            raise ValueError("You must include a path if save argument is True.")
        else:
            path = Path(path)
    _, ax = plt.subplots(figsize=(50, 6))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    date = datetime.datetime.utcfromtimestamp(0)
    start_datetime = date + datetime.timedelta(hours=19)
    end_datetime = date + datetime.timedelta(hours=24+7)
    ax.set_xlim(start_datetime, end_datetime)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(100))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(20))
    ax.tick_params('both', length=20, width=2, which='major')
    ax.tick_params('both', length=20, width=1, which='minor')
    ax.set_ylim(200, 600)
    heights, times = get_heights(resolution=20), get_times(resolution=15)
    _, occurrence_rate, _ = process_season(year=year, month=month, data=data, save=save, path=path)
    plt.pcolor(times, heights, occurrence_rate.T, cmap='jet')
    plt.clim(0, 0.6)
    title = f"Spread F occurrence rate for {month.describe()} - {year}"
    plt.title(title)
    plt.xlabel("Local time")
    plt.ylabel("Height [km]")
    if save:
        plt.savefig(path / f"{title}.pdf")
    logging.info(f'plotted season {year}-{month}')
    # plt.show()


def process(years: List[int], path: Union[Path, str],  save_path: Union[Path, str] = None) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s')
    julia_data: List[pd.DataFrame] = load(path=path, years=years)
    julia_data: pd.DataFrame = pd.concat(julia_data, axis=0, ignore_index=True)
    for year in years: 
        for month in Month:
            plot_season(data=julia_data, year=year, month=month, save=save_path is not None, path=save_path)