import pandas as pd


def UNIX_EPOCH() -> pd.Timestamp:
    return pd.to_datetime(0, unit='s', origin='unix')