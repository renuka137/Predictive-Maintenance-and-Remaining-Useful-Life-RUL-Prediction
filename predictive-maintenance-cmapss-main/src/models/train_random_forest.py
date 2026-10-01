import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import mlflow
import mlflow.sklearn

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100):
    """
    Train Random Forest regressor and evaluate performance.
    """
    mlflow.set_experiment("RUL_RandomForest")

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        print(f"Random Forest RMSE: {rmse:.4f}")
        print(f"Random Forest MAE: {mae:.4f}")

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(model, "model")
        
    return model, rmse, mae
