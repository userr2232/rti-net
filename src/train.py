from numpy.typing import ArrayLike
from omegaconf import DictConfig
import pyarrow as pa
from operator import itemgetter
from typing import Tuple, Optional, Dict
from optuna.trial import Trial
import numpy as np
from pathlib import Path
import logging
import optuna
from torch import optim

from src.folds import fold_loader, load_everything
from src.model import Model, save_jit_model, load_jit_model
from src.engine import Engine
from src.dataset import get_dataloaders


def run_training(cfg: DictConfig, 
                    fold: Tuple[pa.Table, pa.Table], 
                    params: Dict, 
                    trial: Optional[Trial] = None,
                    save_model: bool = False, 
                    prune: bool = True) \
                -> ArrayLike:
    epochs, device, logger_name = itemgetter("epochs", "device", "logger")(cfg.training)
    model = Model(cfg = cfg, 
                    params = params, 
                    trial = trial)
    optimizer = optim.Adam(model.parameters(), 
                            lr = params['initial_lr'])
    engine = Engine(model = model,
                    device = device,
                    optimizer = optimizer)
    best_loss = np.inf
    early_stopping_iter = cfg.training.patience
    early_stopping_counter = 0
    train_table, valid_table = fold
    print("train table shape", train_table.shape)
    print("valid table shape", valid_table.shape)
    train_loader, valid_loader = get_dataloaders(cfg.model.features,
                                                    cfg.model.targets,
                                                    train_table,
                                                    valid_table, 
                                                    scaler_save_path = Path(cfg.model.path) \
                                                        / cfg.model.scaler_checkpoint, 
                                                    **cfg.model.kwargs)
    logger = logging.getLogger(logger_name)
    for epoch in range(epochs):
        train_loss = engine.train(train_loader)
        valid_loss = engine.evaluate(valid_loader)
        logger.info(f"Epoch: {epoch}, \
                        Training Loss: {train_loss}, \
                        Validation Loss: {valid_loss}")
        if round(valid_loss, 3) < round(best_loss, 3):
            early_stopping_counter = 0
            best_loss = valid_loss
            if save_model:
                save_jit_model(cfg = cfg, \
                                model = model)
        else:
            early_stopping_counter += 1
        if prune and early_stopping_counter > early_stopping_iter:
            break
    return best_loss


def train_w_best_params(cfg: DictConfig) -> None:
    study = optuna.load_study(study_name = cfg.sutyd_name, 
                                storage = cfg.hpo.rdb)
    best_trial = study.best_trial

    table = load_everything(cfg)
    num_rows = table.num_rows
    train_pct, valid_pct = itemgetter('train', 'valid')(cfg.final.split)
    train_len, valid_len = num_rows * train_pct // 100, num_rows * valid_pct // 100
    tables = table.slice(0, train_len), table.slice(train_len, valid_len)

    valid_loss = run_training(cfg = cfg,
                                fold = tables, 
                                params = best_trial.params, 
                                save_model = True, 
                                prune = False)
    logger = logging.getLogger(cfg.final.logger)
    logger.info(f"Best trial values: {best_trial.values}")
    logger.info(f"Best trial params: {best_trial.params}")
    logger.info(f"Validation loss: {valid_loss}")


def train_w_best_params(cfg: DictConfig) -> None:
    study = optuna.load_study(study_name = cfg.study_name, 
                                storage = cfg.hpo.rdb)
    best_trial = study.best_trial

    table = load_everything(partitioned_dataset_path = cfg.datasets.partitioned)
    num_rows = table.num_rows
    train_pct, valid_pct = itemgetter('train', 'valid')(cfg.final.split)
    train_len, valid_len = num_rows * train_pct // 100, num_rows * valid_pct // 100
    tables = table.slice(0, train_len), table.slice(train_len, valid_len)
    loss = run_training(cfg = cfg, 
                        fold = tables,
                        params = best_trial.params,
                        save_model = True,
                        prune = False)
    logger = logging.getLogger(cfg.final.logger)
    logger.info(f"Best trial values: {best_trial.values}")
    logger.info(f"Best trial params: {best_trial.params}")
    logger.info(f"Validation loss: {loss}")
