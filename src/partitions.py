from hydra import compose, initialize
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
from operator import itemgetter
import h5py
from omegaconf import DictConfig
import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow import fs
from typing import Dict, Union, Optional
from itertools import product
import os
import re


from src.feature_extraction import extract_features_from_processed_data
from src.preprocessing import preprocessing


def create_partitions(cfg: DictConfig):
    processed_dir = cfg.datasets.processed
    return preprocessing(cfg, save=True, path=Path(processed_dir))

# onset time vs h'F  u  otros parametros geofisicos
# mostrar las correlaciones entre las variables con seaborn
# ver la climatología de los puntos
# hacer los plots de los datos -> deberia haber correspondencia con el paper de chau
# luego lo del modelo
