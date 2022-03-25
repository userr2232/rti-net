import hydra
from omegaconf import DictConfig
from src.preprocessing import read_everything


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    action = cfg.action
    if action == "read_everything":
        print(read_everything(path=cfg.datasets.julia_data, years=range(cfg.data.years.start, cfg.data.years.end+1)))

if __name__ == "__main__":
    main()