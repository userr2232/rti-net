import matplotlib as mpl
from matplotlib import dates as mdates
from matplotlib import (
    pyplot as plt,
    dates as mdates)
from typing import Union, List
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
from numpy.typing import ArrayLike
from matplotlib.axes import Axes
import matplotlib.lines as mlines
import seaborn as sns
from omegaconf import DictConfig
from math import floor


from src.processing import days_of_early_ESF
from src.utils import get_heights, get_times
from src.postprocessing import daily_npys_to_1D_occurrence_df
from src.utils import (
    convert_idx_to_quantity, 
    convert_float_idx_to_quantity,
    convert_float_idx_to_quantity2)
from src.inference import test


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
    rti1 = np.load(rtis_path / ('2017-03-04' + '.npy'))#[:-1,:-1]
    rti2 = np.load(rtis_path / ('2017-03-05' + '.npy'))#[:-1,:-1]
    rti3 = np.load(rtis_path / ('2017-03-06' + '.npy'))#[:-1,:-1]
    rti4 = np.load(rtis_path / ('2017-03-07' + '.npy'))#[:-1,:-1]
    heights = get_heights(resolution=20)
    times1 = get_times(pd.Timestamp('2017-03-04'), resolution=15)
    times2 = get_times(pd.Timestamp('2017-03-05'), resolution=15)
    times3 = get_times(pd.Timestamp('2017-03-06'), resolution=15)
    times4 = get_times(pd.Timestamp('2017-03-07'), resolution=15)
    print("SHAPES:", rti1.shape, times1.shape, heights.shape)
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

def _compute_counts_per_month(df: pd.DataFrame) -> None:
    df = df.loc[(((df.LT.dt.hour == 19)&(df.LT.dt.minute >= 30))|((df.LT.dt.hour == 20)&(df.LT.dt.minute < 30)))].copy()
    df['LT'] = pd.to_datetime(df.LT)
    df['YEAR'] = df.LT.dt.year
    df['MONTH'] = df.LT.dt.month
    df['ESF'] = df.ESF > 0
    df['date'] = df.LT.dt.date
    df['ESF'] = df.groupby('date').ESF.transform(lambda x: x.any())
    df.drop_duplicates('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    counts = df.groupby(['YEAR', 'MONTH']).agg('sum')
    counts.reset_index(inplace=True)
    counts['DAY'] = 1
    counts['date'] = pd.to_datetime(counts.loc[:, ('YEAR', 'MONTH', 'DAY')])
    counts.sort_values('date', inplace=True)
    counts.reset_index(inplace=True, drop=True)
    start_year, start_month = counts.date.iloc[0].date().year, counts.date.iloc[0].date().month
    end_year, end_month = counts.date.iloc[-1].date().year, counts.date.iloc[-1].date().month
    counts_date_range = pd.DataFrame({'date': pd.date_range(pd.Timestamp(year=start_year, month=start_month, day=1), pd.Timestamp(year=end_year, month=end_month, day=1), freq='MS')})
    new_counts = counts_date_range.merge(counts, how='left', on='date')
    new_counts.loc[new_counts['ESF'].isna(), 'ESF'] = 0
    return new_counts

def plot_early_ESF_diff(path: Union[Path,str], ax: Axes = None, set_labels: bool = True, read_npys: bool = False, marker: str='.', color: str='k', markersize: int=1) -> None:
    path = Path(path)
    print("path", path)
    df = None
    baseline_df = daily_npys_to_1D_occurrence_df(path.parent / "snr_20_count_10")
    if read_npys:
        df = daily_npys_to_1D_occurrence_df(path)
    else:
        df = days_of_early_ESF(path=path)
    print("df", df)
    counts = _compute_counts_per_month(df)
    baseline_counts = _compute_counts_per_month(baseline_df)

    counts = counts.merge(baseline_counts, on='date')
    counts['ESF_diff'] = counts.ESF_x - counts.ESF_y
    # print("merged", counts)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(15, 3), tight_layout=True)
    ax.plot(counts.date, counts.ESF_diff, marker=marker, linewidth=0.1, color=color, markersize=markersize, fillstyle='none')
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=range(1, 13, 4)))
    if set_labels:
        ax.set_ylabel('Early ESF counts')
        ax.set_xlabel('Years')
    


def plot_early_ESF_count(path: Union[Path,str], ax: Axes = None, set_labels: bool = True, read_npys: bool = False, marker: str = '.', color: str = 'k', markersize: int = 1) -> None:
    path = Path(path)
    df = None
    if read_npys:
        df = daily_npys_to_1D_occurrence_df(path)
    else:
        df = days_of_early_ESF(path=path)
    counts = _compute_counts_per_month(df)
    print("merged", counts)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(15, 3), tight_layout=True)
    ax.plot(counts.date, counts.ESF, marker=marker, linewidth=0.3, color=color, markersize=markersize, fillstyle='none')
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=range(1, 13, 4)))
    if set_labels:
        ax.set_ylabel('Early ESF counts')
        ax.set_xlabel('Years')
    

def plot_early_ESF_comparison(path: Union[Path,str], snr_thrs: List[int], count_thrs: List[int]) -> None:
    path = Path(path)
    fig, ax = plt.subplots(len(snr_thrs), len(count_thrs), sharex=True, sharey=True, figsize=(15, 4))
    for i, snr in enumerate(snr_thrs):
        for j, count in enumerate(count_thrs):
            if  snr == -20 or count == 10:
                plot_early_ESF_diff(path / f"snr_{-snr}_count_{count}", ax[i][j], set_labels=False, read_npys=True)
                if i == 0: 
                    ax[0][j].set_xlabel(count)
                    ax[0][j].xaxis.set_label_position('top')
        ax[i][0].set_ylabel(snr)
    fig.text(0.5, 0.05, 'Years', ha='center')
    fig.text(0.01, 0.5, 'SNR Threshold', rotation='vertical', va='center')
    fig.text(0.5, 0.86, 'Count Threshold', ha='center')
    plt.subplots_adjust(left=0.055, bottom=0.15, right=0.945, top=0.8, wspace=0.04, hspace=0.1)
    fig.suptitle("ESF occurrences per month w.r.t. to baseline (between 1930 and 2030 LT)")
    plt.show()


def plot_early_ESF_comparison_v2(path: Union[Path,str], snr_thrs: List[int], count_thrs: List[int]) -> None:
    path = Path(path)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 7), tight_layout=True)
    k = 0
    markers = ['P', 'v', 's', 'X']
    colors = ['m', 'g', 'r', 'c']
    snrs_counts = []

    for snr in snr_thrs:
        for count in count_thrs:
            if (snr == -20 or count == 10) and not (snr == -20 and count == 10):
                snrs_counts.append((snr, count))
                plot_early_ESF_diff(path / f"snr_{-snr}_count_{count}", ax[0], set_labels=False, read_npys=True, marker=markers[k], color=colors[k], markersize=4)
                k += 1
    
    legend_markers = [mlines.Line2D([], [], color=color, marker=marker, 
                        linestyle='none', fillstyle='none', markersize=5, label=f"SNR thr.: {snr}, count thr.: {count}") 
                        for color, marker, (snr, count) in zip(colors, markers, snrs_counts)]
    ax[0].legend(handles=legend_markers, fontsize=7.9)
    ax[0].set_ylabel('Occurrences w.r.t baseline')
    ax[1].set_ylabel('Baseline occurrences')
    ax[1].set_xlabel('Months')
    fig.suptitle("ESF occurrences per month between 1930 and 2030 LT")
    plot_early_ESF_count(path / f"snr_20_count_10", ax=ax[1], set_labels=False, read_npys=True)
    plt.show()


def plot_categorical_distribution(data: pd.DataFrame, feature_name: str) -> None:
    ax = sns.countplot(data=data, x=feature_name, palette="ch:.25")
    if "time" in feature_name:
        order = get_times(resolution=15).map(lambda x: x.isoformat()).tolist()
        ax = sns.countplot(data=data, x=feature_name, palette="ch:.25", order=order)
        xticks = ax.get_xticklabels()
        # print(g.ax.get_xticklabels())
        ax.set_xticklabels([pd.to_datetime(text_obj.get_text()).strftime('%H:%M') for text_obj in xticks],rotation=50)
    plt.show()


def plot_categorical_feature_distribution(df: pd.DataFrame, feature_name: str='onset_time_idx') -> None:
    '''
    df: 
        DataFrame that contains the feature to plot
    feature_name:
        str with possible values of: 
        onset_time_idx, end_time_idx, max_height_idx or other name. 
        Note that if it contains "idx" it converts it to a physical quantity
    '''
    # series: pd.Series = df[feature_name]
    if 'idx' in feature_name:
        df[f"{feature_name}_quantity"] = convert_idx_to_quantity(df, feature_name)
    plot_categorical_distribution(df, feature_name=f"{feature_name}_quantity")


def plot_correlations(data: pd.DataFrame) -> None:
    sns.heatmap(data.loc[:, ['V_hF_prev', 'delta_hF_div_delta_time', 
    'foF2', 'V_hF', 'F10.7', 'AP', 'AP (24h)', 'F10.7 (90d dev.)',
    'onset_time_idx', 'end_time_idx', 'max_height_idx']].corr());
    plt.show()


def plot_pred_timeseries(cfg: DictConfig) -> None: # obsolete bc convert_float_idx_to_quantity2 should be used instead
    data = test(cfg)
    targets = cfg.model.targets

    def to_quantity(data: pd.DataFrame, feature_name: str) -> None: # converts to quantity inplace
        data[feature_name] = convert_float_idx_to_quantity(df = data,
                                                            feature_name = feature_name)

    for target in targets:
        print(data[[target, f'{target}_output']])
        to_quantity(data, target)
        to_quantity(data, f'{target}_output')
        data.plot(x = 'LT', y = [target, f'{target}_output'], style='.')
        if 'time' in target:
            print(data[[target, f'{target}_output']])
            formatter = mdates.DateFormatter('%H:%M')
            plt.gca().yaxis.set_major_formatter(formatter)
        plt.show()

        # sampleado por 15 min
        # horas: 7:00, 7:15, 7:30, ....
        # idx: 0 1 2 3 ... 48
        # transform: 0 0.001 0.002 ... 1 f
        # 19:00 + f * 12 * 60 * 60


def plot_pred_timeseries2(cfg: DictConfig) -> None:
    data = test(cfg)
    targets = cfg.model.targets

    data = convert_float_idx_to_quantity2(scaled_df = data,
                                            scaler_path = Path(cfg.model.path) / cfg.model.scaler_checkpoint)

    for target in targets:
        print(data[[target, f'{target}_output']])
        data.plot(x = 'LT', y = [target, f'{target}_output'], style='.')
        if 'time' in target:
            print(data[[target, f'{target}_output']])
            formatter = mdates.DateFormatter('%H:%M')
            plt.gca().yaxis.set_major_formatter(formatter)
        plt.show()
