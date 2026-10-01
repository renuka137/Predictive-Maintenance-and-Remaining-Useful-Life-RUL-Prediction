import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import mlflow
import mlflow.xgboost

def train_xgboost(
    X_train,
    y_train,
    X_test,
    y_test,
    tune_hyperparameters: bool = False,
    n_iter: int = 20,
    random_state: int = 42,
):
    """
    Train XGBoost regressor with optional hyperparameter tuning.
    """
    mlflow.set_experiment("RUL_XGBoost")

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        device="cuda"
    )

    if tune_hyperparameters:
        mlflow.set_experiment("RUL_XGBoost")

        with mlflow.start_run():
            param_grid = {
                "n_estimators": [200, 300],
                "max_depth": [3, 4],
                "learning_rate": [0.01, 0.05],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8],
                "gamma": [0, 0.1, 0.3],
                "reg_alpha": [0, 0.1, 1],
                "reg_lambda": [1, 1.5, 2]
            }

            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring="neg_root_mean_squared_error",
                cv=3,
                verbose=1,
                random_state=random_state,
                n_jobs=-1
            )

            search.fit(X_train, y_train)

            model = search.best_estimator_

            print("Best Parameters:")
            print(search.best_params_)

        # Predictions
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            mlflow.xgboost.autolog()

            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            # Log model
            mlflow.xgboost.log_model(model, "model")

            print(f"XGBoost RMSE: {rmse:.4f}")
            print(f"XGBoost MAE: {mae:.4f}")

    else:
        mlflow.set_experiment("RUL_XGBoost")

        with mlflow.start_run():

            model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
                device="cuda"
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            print("X_test",X_test)

            # Predictions
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            # Log parameters
            mlflow.xgboost.autolog()

            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            # Log model
            mlflow.xgboost.log_model(model, "model")

            print(f"XGBoost RMSE: {rmse:.4f}")
            print(f"XGBoost MAE: {mae:.4f}")

    return model, rmse, mae
