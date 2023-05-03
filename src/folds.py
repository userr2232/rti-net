from omegaconf import DictConfig
from operator import itemgetter
import pyarrow as pa
import pyarrow.dataset as ds
from typing import Iterator, Tuple, Union
from pathlib import Path


def load_dataset(partitioned_dataset_path: Union[str, Path]) \
                -> ds.Dataset:
    partitioned_dir = partitioned_dataset_path
    return ds.dataset(source=partitioned_dir, format="ipc", partitioning="hive")


def year_loader(years: DictConfig, 
                cross_validation: DictConfig) \
                -> Iterator[Tuple[Tuple, Tuple]]:
    START_YEAR, END_YEAR = itemgetter('start', 'end')(years)
    mode, training_window_length, validation_window_length = itemgetter('mode', 'training_window_length', 'validation_window_length')(cross_validation)

    training_start_year = START_YEAR
    if mode == "sliding_window":
        while training_start_year + training_window_length + validation_window_length <= END_YEAR:
            training_end_year = training_start_year + training_window_length
            yield (training_start_year, training_end_year), (training_end_year, training_end_year + validation_window_length)
            training_start_year += 1
    elif mode == "expanding_window":
        while training_start_year + training_window_length + validation_window_length <= END_YEAR:
            training_end_year = training_start_year + training_window_length
            yield (training_start_year, training_end_year), (training_end_year, training_end_year + validation_window_length)
            training_window_length += 1


def fold_loader(partitioned_dataset_path: Union[str, Path],
                years: DictConfig,
                cross_validation: DictConfig) \
                -> Iterator[Tuple[pa.Table, pa.Table]]:

    dataset = load_dataset(partitioned_dataset_path = partitioned_dataset_path)
    loader = year_loader(years = years, 
                            cross_validation = cross_validation)
    for (training_start_year, training_end_year), (validation_start_year, validation_end_year) in loader:
        yield dataset.to_table(filter=((ds.field("year") >= ds.scalar(training_start_year)) & 
                                        (ds.field("year") < ds.scalar(training_end_year)))), \
                dataset.to_table(filter=((ds.field("year") >= ds.scalar(validation_start_year)) &
                                            (ds.field("year") < ds.scalar(validation_end_year))))


def load_everything(partitioned_dataset_path: Union[str, Path]) -> pa.Table:
    return load_dataset(partitioned_dataset_path = partitioned_dataset_path).to_table()