from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/xgb_model.pkl")

WINDOW_SIZE = 30
N_FEATURES = 14
EXPECTED_LENGTH = WINDOW_SIZE * N_FEATURES

class PredictionRequest(BaseModel):
    data: list[float]


@app.post("/predict")
def predict_rul(request: dict):
    data = request["data"]

    expected_features = model.n_features_in_

    if len(data) != expected_features:
        return {
            "error": f"Expected {expected_features} values but got {len(data)}"
        }

    input_array = np.array(data).reshape(1, -1)
    prediction = model.predict(input_array)

    return {"predicted_rul": float(prediction[0])}

