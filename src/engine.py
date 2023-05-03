import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from src.model import Model


class Engine:
    criterion = nn.MSELoss()

    def __init__(self, model: Model, device: Optional[str]="cpu", 
                    optimizer: Optional[Optimizer] = None):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer

    
    def train(self, dataloader: DataLoader) -> float:
        self.model.train()
        final_loss = 0
        for _, (inputs, targets) in dataloader:
            self.optimizer.zero_grad()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = Engine.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(dataloader)

    
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        final_loss = 0
        for _, (inputs, targets) in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = Engine.criterion(outputs, targets)
            final_loss += loss.item()
        return final_loss / len(dataloader)

    
    def test(self, dataloader: DataLoader) -> pd.DataFrame:
        self.model.eval()
        df = pd.DataFrame({})
        for LT, (inputs, targets) in dataloader:
            inputs, targets = inputs.to(self.device).detach(), targets.to(self.device).detach()
            outputs = self.model(inputs).detach()
            df = pd.concat([df, 
                            pd.concat([pd.Series(pd.to_datetime(LT.numpy(), unit = 's')),
                                        pd.DataFrame(np.concatenate((inputs.numpy(), 
                                                                        outputs.numpy(),
                                                                        targets.numpy()),
                                                                        axis = 1))
                                        ],
                                        axis = 1)
                            ])
            df.reset_index(drop = True,
                            inplace = True)
        df.columns = range(df.columns.size)
        return df

