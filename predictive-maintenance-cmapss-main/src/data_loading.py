import pandas as pd
from pathlib import Path


def get_column_names():
    """
    Returns column names for CMAPSS dataset.
    """
    cols = ["unit_id", "cycle"]

    # 3 operating settings
    cols += [f"op_setting_{i}" for i in range(1, 4)]

    # 21 sensor measurements
    cols += [f"sensor_{i}" for i in range(1, 22)]

    return cols


def load_train_data(data_path: str) -> pd.DataFrame:
    """
    Load training data.
    """
    columns = get_column_names()

    df = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None,
        names=columns
    )

    return df


def load_test_data(data_path: str) -> pd.DataFrame:
    """
    Load test data.
    """
    columns = get_column_names()

    df = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None,
        names=columns
    )

    return df


def load_rul_data(data_path: str) -> pd.DataFrame:
    """
    Load RUL values for test set.
    """
    df = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None,
        names=["RUL"]
    )

    return df
