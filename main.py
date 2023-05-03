# como comenzamos a colaborar
# como se daba las colaboraciones
# pros y contras
# enfasis en poner deadlines con conferencias y eventos
# enfasis en colaboraciones remotas

import hydra
from omegaconf import DictConfig
from pathlib import Path

from src.preprocessing import load, preprocessing
from src.processing import process
from src.plots import (
    inputs_plot,
    zoomed_plot,
    plot_early_ESF_count,
    plot_early_ESF_comparison,
    plot_early_ESF_comparison_v2,
    plot_correlations,
    plot_categorical_feature_distribution,
    plot_pred_timeseries,
    plot_pred_timeseries2)
from src.feature_extraction import (
    extract_features, 
    extract_features_from_processed_data)
from src.partitions import create_partitions
from src.optimization import run_study
from src.train import train_w_best_params
from src.inference import test


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    action = cfg.action
    if action == "load":
        print(load(path = cfg.datasets.julia, 
                    years = range(cfg.data.years.start, cfg.data.years.end+1)))
    elif action == "process":
        process(years = range(cfg.data.years.start, cfg.data.years.end+1), 
                path = cfg.datasets.julia_data, 
                save_path = cfg.datasets.processed, 
                snr_thr = cfg.thr.snr, 
                count_thr = cfg.thr.count)
    elif action == "extract_features":
        print(extract_features(load(path = cfg.datasets.julia_data, 
                                    years = range(cfg.data.years.start, cfg.data.years.end+1))))
    elif action == "extract_features_from_processed_data":
        print(extract_features_from_processed_data(path = cfg.datasets.processed))
    elif action == "inputs_plot":
        inputs_plot(path = cfg.datasets.geo_param)
    elif action == "zoomed_plot":
        zoomed_plot(geo_path = cfg.datasets.geo_param, 
                    rtis_path = cfg.datasets.processed)
    elif action == "plot_early_ESF_count":
        plot_early_ESF_count(path = cfg.datasets.julia_data)
    elif action == "plot_early_ESF_comparison":
        plot_early_ESF_comparison(path = Path(cfg.root) / "thresholds", 
                                    snr_thrs = [-10, -20, -40], 
                                    count_thrs = [5, 10, 20])
    elif action == "plot_early_ESF_comparison_v2":
        plot_early_ESF_comparison_v2(path = Path(cfg.root) / "thresholds", 
                                     snr_thrs = [-10, -20, -40], 
                                     count_thrs = [5, 10, 20])
    elif action == "extract_features_from_processed_data":
        print(extract_features_from_processed_data(path = Path(cfg.datasets.processed)))
    elif action == "create_partitions":
        create_partitions(cfg)
    elif action == "preprocessing":
        print(preprocessing(cfg = cfg, 
                            path = Path(cfg.datasets.processed)).to_pandas())
    elif action == "plot_targets":
        df = extract_features_from_processed_data(path = cfg.datasets.processed)
        print("extracted features", df)
        feature_names = cfg.model.targets
        for feature_name in feature_names:
            plot_categorical_feature_distribution(df = df, 
                                                    feature_name = feature_name)
    elif action == "plot_correlations":
        data = preprocessing(cfg = cfg, 
                                path = Path(cfg.datasets.processed)).to_pandas()
        plot_correlations(data = data)
    elif action == "run_study":
        run_study(cfg)
    elif action == "train_w_best_params":
        train_w_best_params(cfg)
    elif action == "test":
        test(cfg)
    elif action == "plot_pred_timeseries":
        plot_pred_timeseries(cfg)
    elif action == "plot_pred_timeseries2":
        plot_pred_timeseries2(cfg)
if __name__ == "__main__":
    main()