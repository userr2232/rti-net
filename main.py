import hydra
from omegaconf import DictConfig
from src.preprocessing import load
from src.processing import process


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    action = cfg.action
    if action == "load":
        print(load(path=cfg.datasets.julia_data, years=range(cfg.data.years.start, cfg.data.years.end+1)))
    elif action == "process":
        process(years=range(cfg.data.years.start, cfg.data.years.end+1), path=cfg.datasets.julia_data, save_path=cfg.datasets.processed)

if __name__ == "__main__":
    main()