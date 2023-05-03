from omegaconf import DictConfig
from operator import itemgetter
from pathlib import Path

from src.folds import load_everything
from src.dataset import get_dataloaders
from src.model import load_jit_model
from src.engine import Engine
from src.postprocessing import inverse_transform


def test(cfg: DictConfig, invert: bool = False) -> None:
    train_pct, valid_pct = itemgetter('train', 'valid')(cfg.final.split)
    test_pct = 100 - train_pct - valid_pct
    table = load_everything(cfg.datasets.partitioned)
    num_rows = table.num_rows
    test_offset = num_rows * (100 - test_pct) // 100
    test_table = table.slice(test_offset)
    test_loader = get_dataloaders(cfg.model.features, 
                                    cfg.model.targets,
                                    test_table, 
                                    scaler = Path(cfg.model.path) / cfg.model.scaler_checkpoint)
    model = load_jit_model(cfg)
    
    engine = Engine(model=model)
    scaled_df = engine.test(test_loader)

    features = sorted(cfg.model.features)
    targets = sorted(cfg.model.targets)

    target_outputs = [ f'{target}_output' for target in targets ]
    scaled_df.columns = ['LT'] + features + target_outputs + targets
    df = scaled_df.copy()
    if invert:
        print("receiving columns", len(sorted(features + targets)))
        print("receiving output columns", len(sorted(features + target_outputs)))
        DOY = ['DNC', 'DNS']
        filter_doy = lambda x: x not in DOY
        df.loc[:, sorted(list(
                        filter(filter_doy, 
                                features + targets)))] = inverse_transform(df = scaled_df,
                                                                            scaler = Path(cfg.model.path) / cfg.model.scaler_checkpoint)
        df.loc[:, sorted(list(
                        filter(filter_doy, 
                                features + target_outputs)))] = inverse_transform(df = scaled_df,
                                                                                    scaler = Path(cfg.model.path) / cfg.model.scaler_checkpoint,
                                                                                    outputs = True)
    return df
