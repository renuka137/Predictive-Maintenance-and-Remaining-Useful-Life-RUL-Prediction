# src/train.py
import argparse
from src.pipeline import build_dataset
from src.models.train_lstm import train_lstm
from src.models.train_xgboost import train_xgboost
from src.models.train_random_forest import train_random_forest

import joblib
import torch

def main(model_type, train_path, test_path, rul_path):

    lstm = model_type.lower() == "lstm"

    X_train, X_test, y_train, y_test, scaler = build_dataset(
        train_path, test_path, rul_path, lstm=lstm
    )

    if model_type.lower() == "xgb":
        model, rmse, mae = train_xgboost(X_train, y_train, X_test, y_test)
        joblib.dump(model, "models/xgb_model.pkl")
    elif model_type.lower() == "rf":
        model, rmse, mae = train_random_forest(X_train, y_train, X_test, y_test)
        joblib.dump(model, "models/rf_model.pkl")
    elif model_type.lower() == "lstm":
        model, rmse, mae, _ = train_lstm(X_train, y_train, X_test, y_test)
        torch.save(model, "models/lstm_model.pt")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"{model_type} training complete. RMSE={rmse:.4f}, MAE={mae:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="xgb | rf | lstm")
    parser.add_argument("--train_path", type=str, default="data/raw/train_FD001.txt")
    parser.add_argument("--test_path", type=str, default="data/raw/test_FD001.txt")
    parser.add_argument("--rul_path", type=str, default="data/raw/RUL_FD001.txt")
    args = parser.parse_args()

    main(args.model, args.train_path, args.test_path, args.rul_path)
