# import torch
# from torch import nn, optim
# import torchvision
# from torchvision import datasets
# from torch.utils.data import transforms as T
# from torch.utils.tensorboard import SummaryWriter


# class Discriminator(nn.Module):
#     def __init__(self, in_features):
#         super().__init__()
#         self.disc = nn.Sequential(
#             nn.Linear(in_features, 128),
#             nn.LeakyReLU(0.1),
#             nn.Linear(128, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         return self.disc(x)


# class Generator(nn.Module):
#     def __init__(self, z_dim, img_dim):
        
from __future__ import annotations
import torch
import torch.nn as nn
from enum import Enum
from omegaconf import DictConfig
from typing import List, Optional, Dict, Any, Union
from optuna.trial import Trial
from operator import itemgetter
import re
from pathlib import Path
from torch.jit import ScriptModule
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


class Activation(Enum):
    ELU = nn.ELU()
    LeakyReLU = nn.LeakyReLU() 
    PReLU = nn.PReLU()
    ReLU = nn.ReLU()
    RReLU = nn.RReLU()
    SELU = nn.SELU()
    CELU = nn.CELU()

    @classmethod
    def builder(cls: Activation, name: str) -> nn.Module:
        return cls.__members__[name].value

    @staticmethod
    def types() -> List[str]:
        return ['ELU', 'LeakyReLU', 'ReLU', 'RReLU', 'SELU', 'CELU']


class Model(nn.Module):
    def __init__(self, cfg: DictConfig, params: Optional[Dict], trial: Optional[Trial] = None) -> None:
        super().__init__()
        features, targets = itemgetter("features", "targets")(cfg.model)
        ntargets = len(targets)
        nfeatures = len(features)
        self.trial = trial
        self.params = params
        self.cfg = cfg
        activation = self.param_getter("activation")
        in_features = nfeatures
        nlayers = self.param_getter("nlayers")
        layers = []
        for i in range(nlayers):
            out_features = self.param_getter(f"n_units_l{i}")
            layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
            layers.append(activation)
            p = self.param_getter(f"dropout_l{i}") # Dropout every layer?
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(in_features=in_features, out_features=ntargets))
        self.model = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def param_getter(self, param_name: str) -> Any:
        min_nlayers, max_nlayers = itemgetter('min_nlayers', 'max_nlayers')(self.cfg.hpo)
        min_nunits, max_nunits = itemgetter('min_nunits', 'max_nunits')(self.cfg.hpo)
        min_dropout, max_dropout = itemgetter('min_dropout', 'max_dropout')(self.cfg.hpo)
        
        def trial_suggest(param_name: str) -> Any:
            if param_name == "activation":
                return Activation.builder(
                    self.trial.suggest_categorical(
                        param_name, 
                        Activation.types()))
            elif param_name == "nlayers":
                return self.trial.suggest_int(
                    param_name,
                    min_nlayers,
                    max_nlayers)
            elif re.match(r"^n_units_l\d+$", param_name):
                return self.trial.suggest_int(
                    param_name,
                    min_nunits,
                    max_nunits)
            elif re.match(r"^dropout_l\d+$", param_name):
                return self.trial.suggest_float(
                    param_name,
                    min_dropout,
                    max_dropout)
            else:
                raise ValueError("Invalid parameter name.")
        try:
            return self.params[param_name] if param_name != "activation" else Activation.builder(self.params[param_name])
        except KeyError:
            return trial_suggest(param_name)


def save_jit_model(cfg: DictConfig, model: nn.Module) -> None:
    path = Path(cfg.model.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_model = model.cpu()
    sample_input_cpu = torch.rand(len(cfg.model.features))
    traced_cpu = torch.jit.trace(cpu_model, sample_input_cpu)
    torch.jit.save(traced_cpu, Path(cfg.model.path) / cfg.model.nn_checkpoint)


def load_jit_model(cfg: DictConfig) -> ScriptModule:
    path = Path(cfg.model.path)
    return torch.jit.load(path / cfg.model.nn_checkpoint)


class Scaler(MinMaxScaler):
    def __init__(self, df: pd.DataFrame, columns: List[str]): # columns should be sorted
        super().__init__() # MinMaxScaler by default scales from 0 to 1
        self.columns = columns
        print("fit columns", self.columns)
        self.fit(df.loc[:, columns])
    
    def save(self, path: Union[Path,str]) -> None:
        path = Path(path)
        if not path.parent.exists():
            Path.mkdir(path.parent)
        joblib.dump(self, path)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print("transform columns", self.columns)
        return super().transform(df.loc[:, self.columns])

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return super().inverse_transform(df.loc[:, self.columns])

    @staticmethod
    def load(path: Union[Path,str]) -> Scaler:
        return joblib.load(path)
