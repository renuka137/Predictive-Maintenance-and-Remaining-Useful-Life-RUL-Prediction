import numpy as np
import pandas as pd


def create_window_features(df: pd.DataFrame, window_size: int = 30):
    """
    Create window-based features for supervised learning.
    """
    feature_cols = [
        col for col in df.columns
        if col not in ["unit_id", "cycle", "RUL"]
    ]

    X = []
    y = []

    for unit in df["unit_id"].unique():
        unit_data = df[df["unit_id"] == unit]

        for i in range(window_size, len(unit_data)):
            window = unit_data.iloc[i-window_size:i]

            # Flatten window values
            features = window[feature_cols].values.flatten()

            X.append(features)
            y.append(unit_data.iloc[i]["RUL"])

    X = np.array(X)
    y = np.array(y)

    return X, y


def create_lstm_sequences(df: pd.DataFrame, window_size: int = 30):
    """
    Create 3D sequences for LSTM.
    Output shape:
    X -> (samples, window_size, features)
    y -> (samples,)
    """

    feature_cols = [
        col for col in df.columns
        if col not in ["unit_id", "cycle", "RUL"]
    ]

    X = []
    y = []

    for unit in df["unit_id"].unique():
        unit_data = df[df["unit_id"] == unit]

        for i in range(window_size, len(unit_data)):
            window = unit_data.iloc[i-window_size:i]

            X.append(window[feature_cols].values)
            y.append(unit_data.iloc[i]["RUL"])

    return np.array(X), np.array(y)
