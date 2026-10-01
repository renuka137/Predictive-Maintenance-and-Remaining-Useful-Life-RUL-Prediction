# src/pipeline.py
from src.data_loading import load_train_data, load_test_data, load_rul_data
from src.preprocessing import compute_train_rul, compute_test_rul, cap_rul, remove_constant_columns, scale_features
from src.feature_engineering import create_window_features, create_lstm_sequences
import pandas as pd

def build_dataset(train_path, test_path, rul_path, cap=125, lstm=False, window_size=30):
    """
    Load, preprocess, and feature engineer the CMAPSS dataset.
    
    Returns:
        X_train, X_test, y_train, y_test
        (or 3D arrays if lstm=True)
    """
    #  Load data
    train_df = load_train_data(train_path)
    test_df = load_test_data(test_path)
    rul_df = load_rul_data(rul_path)

    #  Preprocess
    train_df = compute_train_rul(train_df)
    test_df = compute_test_rul(test_df, rul_df)

    train_df = cap_rul(train_df, cap)
    test_df = cap_rul(test_df, cap)

    train_df = remove_constant_columns(train_df)
    test_df = remove_constant_columns(test_df)

    train_df, test_df, scaler = scale_features(train_df, test_df)

    #  Feature engineering
    if lstm:
        X_train, y_train = create_lstm_sequences(train_df, window_size)
        X_test, y_test = create_lstm_sequences(test_df, window_size)
    else:
        X_train, y_train = create_window_features(train_df, window_size)
        X_test, y_test = create_window_features(test_df, window_size)

    return X_train, X_test, y_train, y_test, scaler
