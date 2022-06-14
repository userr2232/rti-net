import matplotlib as mpl
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from typing import Union
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
from src.processing import get_heights, get_times
from numpy.typing import ArrayLike
from matplotlib.axes import Axes

plt.rcParams.update({"font.family": "serif", "font.serif": ["Palatino"]})


def inputs_plot(path: Union[Path,str]) -> None:
    h5_d = h5py.File(path, 'r')
    sao = pd.DataFrame(h5_d['Data']['SAO_total'][()])
    geo_param = pd.DataFrame(h5_d['Data']['GEO_param'][()])
    geo_param.loc[:, 'datetime'] = pd.to_datetime(geo_param.loc[:, ('YEAR', 'MONTH', 'DAY', 'HOUR')]) - pd.Timedelta(5, unit='hours')
    sao.rename(columns={'MIN': 'MINUTE', 'SEC': 'SECOND'}, inplace=True)
    sao.loc[:, 'datetime'] = pd.to_datetime(sao.loc[:, ('YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE')]) - pd.Timedelta(5, unit='hours')
    sao.sort_values('datetime', inplace=True)
    sao.reset_index(inplace=True)
    date_range = pd.Series(pd.to_datetime(pd.date_range(start='2000-01-01', end='2021-12-31', freq='24H')) + pd.Timedelta(19, unit='hours') + pd.Timedelta(30, unit='minutes'), name="datetime")
    sao_datetime_idx = sao.copy()
    sao = pd.merge_asof(date_range, sao, on='datetime', tolerance=pd.Timedelta("1min"))
    date_range_15 = date_range - pd.Timedelta(15, unit='minutes')
    date_range_30 = date_range - pd.Timedelta(30, unit='minutes')
    sao_datetime_idx = pd.merge_asof(pd.concat([date_range, date_range_15, date_range_30]).sort_values(), 
                                        sao_datetime_idx, on='datetime', tolerance=pd.Timedelta("1min"))
    sao_datetime_idx['date'] = sao_datetime_idx.datetime.dt.date
    geo_param = pd.merge_asof(date_range, geo_param, on='datetime', tolerance=pd.Timedelta("30min"))
    geo_param_datetime_idx = geo_param.copy()
    geo_param_datetime_idx.index = pd.to_datetime(geo_param_datetime_idx.datetime)
    geo_param_datetime_idx['F10.7 (90d)'] = geo_param_datetime_idx['F10.7'].rolling('90d').mean()
    geo_param_datetime_idx['AP (24h)'] = geo_param_datetime_idx['AP'].rolling('24h').mean()
    sao_datetime_idx.index = pd.to_datetime(sao_datetime_idx.datetime)
    sao_datetime_idx['V_hF_prev'] = sao_datetime_idx['V_hF'].rolling('30min').agg(lambda rows: rows[0])
    sao_datetime_idx['V_hF_prev_time'] = sao_datetime_idx['V_hF'].rolling('30min').agg(lambda rows: pd.to_datetime(rows.index[0]).value)
    sao_datetime_idx['V_hF_prev_time'] = pd.to_datetime(sao_datetime_idx['V_hF_prev_time'])
    sao_datetime_idx['delta_hF'] = sao_datetime_idx['V_hF']-sao_datetime_idx['V_hF_prev']
    sao_datetime_idx['delta_time'] = (sao_datetime_idx['datetime']-sao_datetime_idx['V_hF_prev_time']).dt.components.minutes + 1e-9
    sao_datetime_idx['delta_hF_div_delta_time'] = sao_datetime_idx['delta_hF'] / sao_datetime_idx['delta_time']
    sao_datetime_idx['date'] = sao_datetime_idx.datetime.dt.date
    sao_datetime_idx.drop_duplicates('date', keep='last', inplace=True)
    _, ax = plt.subplots(9, 1, sharex=True, tight_layout=True, figsize=(15,10))
    ax[0].plot(sao.datetime, sao.V_hF, marker='.', linewidth=0, markersize=0.5, color='darkslategray')
    ax[0].set_ylabel("h'F")
    ax[1].plot(sao_datetime_idx.datetime, sao_datetime_idx['V_hF_prev'], marker='.', linewidth=0, markersize=0.5, color='darkslategray')
    ax[1].set_ylabel("prev. h'F")
    ax[2].plot(sao_datetime_idx.datetime, sao_datetime_idx['delta_hF'], marker='.', linewidth=0, markersize=0.5, color='darkslategray')
    ax[2].set_ylabel("∆h'F")
    ax[2].set_ylim(-100, 100)
    ax[3].plot(sao_datetime_idx.datetime, sao_datetime_idx['delta_hF_div_delta_time'], marker='.', linewidth=0, markersize=0.5, color='darkslategray')
    ax[3].set_ylabel("∆h'F/∆t")
    ax[3].set_ylim(-6, 6)
    ax[4].plot(sao.datetime, sao.foF2, marker='.', linewidth=0, markersize=0.5, color='peru')
    ax[4].set_ylabel("foF2")
    ax[5].plot(geo_param.datetime, geo_param['F10.7'], marker='.', linewidth=0, markersize=0.5, color='gold')
    ax[5].set_ylabel("F10.7")
    ax[6].plot(geo_param_datetime_idx.datetime, geo_param_datetime_idx['F10.7 (90d)'], marker='.', linewidth=0, markersize=0.5, color='gold')
    ax[6].set_ylabel("F10.7 (90d)")
    ax[7].stackplot(geo_param_datetime_idx.datetime, geo_param_datetime_idx['AP'], linewidth=0.5, color='royalblue')
    ax[7].set_ylabel("ap")
    ax[8].stackplot(geo_param_datetime_idx.datetime, geo_param_datetime_idx['AP (24h)'], linewidth=0.5, color='royalblue')
    ax[8].set_ylabel("ap (24h)")
    ax[8].xaxis.set_major_locator(mdates.YearLocator())
    plt.show()


def plot_compacted_spreadF(spreadF_map: ArrayLike, times: ArrayLike, heights: ArrayLike, ax: Axes, plot_rti: bool = False) -> None:
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    collapsed = spreadF_map.sum(axis=1) > 0
    _ = ax.pcolormesh(times, np.array([1,2]), np.concatenate([collapsed, collapsed]).reshape(2,-1))
    if not plot_rti:
        ax.set_ylabel("Occurrence")


def plot_2d_spreadF(spreadF_map: ArrayLike, times: ArrayLike, heights: ArrayLike, ax: Axes) -> None:
    ax.pcolor(times, heights, spreadF_map.T)
    ax.set_ylabel("height [km]")


def zoomed_plot(geo_path: Union[Path,str], rtis_path: Union[Path,str], plot_rti: bool = True) -> None:
    geo_path, rtis_path = Path(geo_path), Path(rtis_path)
    h5_d = h5py.File(geo_path, 'r')
    sao = pd.DataFrame(h5_d['Data']['SAO_total'][()])
    sao.rename(columns={'MIN': 'MINUTE', 'SEC': 'SECOND'}, inplace=True)
    sao.loc[:, 'datetime'] = pd.to_datetime(sao.loc[:, ('YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE')]) - pd.Timedelta(5, unit='hours')
    sao.sort_values('datetime', inplace=True)
    sao.reset_index(inplace=True)
    _, ax = plt.subplots(2 + (int(plot_rti)), 1, sharex=True, tight_layout=True, figsize=(15, 4), gridspec_kw={'height_ratios': [1, 3, 9] if plot_rti else [1, 4]})
    sao = sao.loc[((sao.datetime >= pd.to_datetime('2017-03-05'))&(sao.datetime <= pd.to_datetime('2017-03-08')))].copy()
    rti1 = np.load(rtis_path / ('2017-03-04' + '.npy'))
    rti2 = np.load(rtis_path / ('2017-03-05' + '.npy'))
    rti3 = np.load(rtis_path / ('2017-03-06' + '.npy'))
    rti4 = np.load(rtis_path / ('2017-03-07' + '.npy'))
    heights = get_heights(resolution=20)
    times1 = get_times(pd.Timestamp('2017-03-04'), resolution=15)
    times2 = get_times(pd.Timestamp('2017-03-05'), resolution=15)
    times3 = get_times(pd.Timestamp('2017-03-06'), resolution=15)
    times4 = get_times(pd.Timestamp('2017-03-07'), resolution=15)
    plot_compacted_spreadF(spreadF_map=rti1, times=times1, heights=heights, ax=ax[0], plot_rti=plot_rti)
    plot_compacted_spreadF(spreadF_map=rti2, times=times2, heights=heights, ax=ax[0], plot_rti=plot_rti)
    plot_compacted_spreadF(spreadF_map=rti3, times=times3, heights=heights, ax=ax[0], plot_rti=plot_rti)
    plot_compacted_spreadF(spreadF_map=rti4, times=times4, heights=heights, ax=ax[0], plot_rti=plot_rti)
    if plot_rti:
        plot_2d_spreadF(spreadF_map=rti1, times=times1, heights=heights, ax=ax[1])
        plot_2d_spreadF(spreadF_map=rti2, times=times2, heights=heights, ax=ax[1])
        plot_2d_spreadF(spreadF_map=rti3, times=times3, heights=heights, ax=ax[1])
        plot_2d_spreadF(spreadF_map=rti4, times=times4, heights=heights, ax=ax[1])
    ax[-1].plot(sao.datetime, sao.V_hF, marker='.', linewidth=0, color='black')
    ax[-1].xaxis.set_major_locator(mdates.DayLocator())
    ax[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    ax[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    ax[-1].tick_params('x', length=20, width=1.2, which='major')
    ax[-1].tick_params('both', length=10, width=1, which='minor')
    ax[-1].set_ylabel("h'F [km]")
    plt.xlabel("Local time")
    plt.xlim(pd.to_datetime('2017-03-05'), pd.to_datetime('2017-03-08'))
    plt.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.2)
    plt.grid(b=True, which='minor', color='k', linestyle='--', linewidth=0.1)
    plt.show()