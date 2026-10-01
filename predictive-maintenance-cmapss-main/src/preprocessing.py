import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for training data.
    """
    max_cycle = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycle.columns = ["unit_id", "max_cycle"]

    df = df.merge(max_cycle, on="unit_id")

    df["RUL"] = df["max_cycle"] - df["cycle"]

    df.drop(columns=["max_cycle"], inplace=True)

    return df


def compute_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RUL for test data using provided RUL file.
    """
    max_cycle = test_df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycle.columns = ["unit_id", "max_cycle"]

    # RUL file is ordered by engine_id
    max_cycle["RUL_given"] = rul_df["RUL"].values

    test_df = test_df.merge(max_cycle, on="unit_id")

    test_df["RUL"] = (
        test_df["RUL_given"]
        + (test_df["max_cycle"] - test_df["cycle"])
    )

    test_df.drop(columns=["max_cycle", "RUL_given"], inplace=True)

    return test_df


def cap_rul(df: pd.DataFrame, threshold: int = 125) -> pd.DataFrame:
    """
    Cap RUL values to avoid very large early-life targets.
    """
    df["RUL"] = df["RUL"].clip(upper=threshold)
    return df


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns with zero variance.
    """
    constant_cols = [
        col for col in df.columns
        if df[col].nunique() <= 1
    ]

    df = df.drop(columns=constant_cols)

    return df


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Scale sensor features using StandardScaler.
    """
    feature_cols = [
        col for col in train_df.columns
        if col not in ["unit_id", "cycle", "RUL"]
    ]

    scaler = StandardScaler()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df, scaler
