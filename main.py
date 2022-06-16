import hydra
from omegaconf import DictConfig
from src.preprocessing import load
from src.processing import process
from src.plots import inputs_plot, zoomed_plot, plot_early_ESF_count, plot_early_ESF_comparison
from pathlib import Path


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    action = cfg.action
    if action == "load":
        print(load(path=cfg.datasets.julia_data, years=range(cfg.data.years.start, cfg.data.years.end+1)))
    elif action == "process":
        process(years=range(cfg.data.years.start, cfg.data.years.end+1), path=cfg.datasets.julia_data, save_path=cfg.datasets.processed, 
                snr_thr=cfg.thr.snr, count_thr=cfg.thr.count)
    elif action == "inputs_plot":
        inputs_plot(path=cfg.datasets.geo_param)
    elif action == "zoomed_plot":
        zoomed_plot(geo_path=cfg.datasets.geo_param, rtis_path=cfg.datasets.processed)
    elif action == "plot_early_ESF_count":
        plot_early_ESF_count(path=cfg.datasets.julia_data)
    elif action == "plot_early_ESF_comparison":
        plot_early_ESF_comparison(path=Path(cfg.root) / "thresholds", snr_thrs=[-10, -20, -40], count_thrs=[5, 10, 20])

if __name__ == "__main__":
    main()