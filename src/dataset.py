from typing import List, Optional, Union
import pyarrow as pa
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


from src.processing import process_dataframe
from src.model import Scaler


class RTIDataset(Dataset):
    def __init__(self, 
                    features: List[str], 
                    targets: List[str], 
                    table: pa.Table, 
                    scaler: Optional[Union[Scaler, str, Path]] = None, 
                    scaler_save_path: Optional[Union[Path, str]] = None) \
                -> None:
        self.df, self.scaler = process_dataframe(columns = features + targets, 
                                                    df = table.to_pandas(),
                                                    scaler = scaler, 
                                                    scaler_save_path = scaler_save_path)
        
        self.LT = self.df.loc[:, 'LT-1h']
        features = sorted(features)
        targets = sorted(targets)
        print("features", features)
        print("targets", targets)
        self.features_df = self.df.loc[:, features]
        self.targets_df = self.df.loc[:, targets]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        return np.datetime64(self.LT.iloc[idx]).astype('datetime64[s]').astype('int'), \
                    (torch.tensor(self.features_df.iloc[idx], 
                                    dtype = torch.float32,
                                    requires_grad = True), \
                    torch.tensor(self.targets_df.iloc[idx], 
                                    dtype = torch.float32,
                                    requires_grad = True))

def get_dataloaders(features: List[str],
                    targets: List[str], 
                    *args: pa.Table, 
                    scaler: Optional[Union[Scaler, Path, str]] = None, 
                    scaler_save_path: Optional[Union[Path, str]] = None, 
                    **kwargs) \
                    -> Union[List[DataLoader], DataLoader]:
    if len(args) == 1:
        return DataLoader(dataset = RTIDataset(features = features,
                                                targets = targets,
                                                table = args[0],
                                                scaler = scaler,
                                                scaler_save_path = scaler_save_path))
    else:
        training_dataset = RTIDataset(features = features,
                                        targets = targets,
                                        table = args[0],
                                        scaler = scaler,
                                        scaler_save_path = scaler_save_path)
        scaler: Scaler = training_dataset.scaler
        datasets = [ training_dataset ] + [ RTIDataset(features = features,
                                                        targets = targets,
                                                        table = table,
                                                        scaler = scaler,
                                                        scaler_save_path = scaler_save_path)
                                                            for table in args[1:] ]
        return [ DataLoader(dataset=dataset, **kwargs) 
                    for dataset in datasets ]