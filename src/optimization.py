from omegaconf import DictConfig
import optuna
from operator import itemgetter
from functools import partial
from optuna.trial import Trial
from numpy.typing import ArrayLike
from typing import List, Tuple, Optional
import numpy as np
import pyarrow as pa
import logging

from src.folds import fold_loader
from src.train import run_training


def objective(trial: Trial,
                cfg: DictConfig) \
                -> ArrayLike:
    min_lr, max_lr = itemgetter('min_lr', 'max_lr')(cfg.hpo)
    params = {
        "initial_lr": trial.suggest_loguniform("initial_lr", min_lr, max_lr)
    }
    losses = []
    loader = fold_loader(partitioned_dataset_path = cfg.datasets.partitioned,
                            years = cfg.data.years,
                            cross_validation = cfg.cross_validation)
    for fold in loader:
        loss = run_training(cfg=cfg, fold=fold, params=params, trial=trial)
        losses.append(loss)
    return np.mean(losses)


def run_study(cfg: DictConfig) -> None:
    study = optuna.create_study(study_name = cfg.study_name, 
                                storage = cfg.hpo.rdb, 
                                direction = 'minimize', 
                                load_if_exists = True)
    ntrials, logger_name = itemgetter('ntrials', 'logger')(cfg.hpo)
    study.optimize(partial(objective, cfg=cfg), n_trials=ntrials)

    best_trial = study.best_trial

    logger = logging.getLogger(logger_name)
    logger.info(f"Best trial values: {best_trial.values}")
    logger.info(f"Best trial params: {best_trial.params}")


def view_study(cfg: DictConfig) -> None:
    study = optuna.load_study(study_name=cfg.study_name, storage=cfg.hpo.rdb)
    best_trial = study.best_trial
    print(best_trial)

    fig = optuna.visualization.plot_parallel_coordinate(study, params=["activation", "nlayers", "initial_lr"], target_name="Loss")
    fig.write_html(cfg.hpo.plot)
