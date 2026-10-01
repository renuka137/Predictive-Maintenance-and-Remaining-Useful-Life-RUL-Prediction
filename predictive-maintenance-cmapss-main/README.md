# 🛠️ Predictive Maintenance using NASA C-MAPSS
> ML & DL for Turbofan Engine Remaining Useful Life (RUL) Prediction

---

## 🚀 Overview

This project builds a complete predictive maintenance pipeline using the real-world **NASA C-MAPSS** dataset.
- Goal: Predict the Remaining Useful Life (RUL) of aircraft engines before failure, enabling cost-saving maintenance decisions.

---

## 📦 Dataset

- **Source:** [NASA Turbofan Engine Degradation Simulation Data Set](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)  
- **About:** Four subsets of time-series sensor data (FD001-FD004) with different operating and fault conditions.  
  Each row = measurements for an engine at a given cycle.
- **Columns:** Engine ID, time cycle, 3 operational settings, 21 sensor measurements, and RUL label (for test data).

---

## 🧩 Solution Architecture

- **Data Processing:** Feature engineering, normalization, sequence generation (for LSTMs/CNNs)
- **Modeling Approaches:**
  - ML: Random Forest, XGBoost, SVR, Gradient Boosting
  - DL: RNN/LSTM, CNN-LSTM hybrids for time series
- **Evaluation:** RMSE, MAE, NASA scoring
- **Deployment:** FastAPI endpoint for real-time RUL prediction
- **Packaging:** Docker for reproducible infra & fast local/cloud deployment

---

## 💻 Notebooks & Script Index

- `notebooks/eda_cmapss.ipynb` - Exploratory analysis, feature trends, RUL histograms
- `notebooks/ml_baselines.ipynb` - RandomForest/XGBoost, grid search
- `notebooks/lstm_training.ipynb` - Deep learning for time series RUL prediction
- `src/app.py` - FastAPI deployment script
- `Dockerfile` - Containerization for scalable serving

---

## 📈 Key Results & Demo

- **Best Model:** LSTM (window 50)  
  - RMSE (FD001 test): **22.8 cycles**
  - NASA Score (PHM08): **345** (lower = better)
- **RandomForest Baseline:** RMSE ~32.0, quick setup, strong baseline
- **Top Features:** Sensor 11 (Temperature), Sensor 15 (Fan Vibration), operational settings

| Model         | RMSE (FD001) | NASA Score |
|---------------|-------------|------------|
| RandomForest  |    32.0     |   520      |
| XGBoost       |    28.2     |   412      |
| LSTM          |    22.8     |   345      |

---

## 🚦 Example API Usage

You can run the RUL prediction web service locally:

```bash
docker build -t rul-service .
docker run -p 8000:8000 rul-service
```

Example POST request:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"engine_id":1, "cycle":165, "sensor1":..., ...}'
```

---

## 🎯 Insights

- Sequence-based DL models outperform traditional ML for predictive maintenance on sensor time-series.
- Early warning on RUL > enables planned maintenance and reduced operational cost.
- Explainability analysis revealed temperature and vibration sensors as key predictors.

---

## 📚 References
- NASA Data Page: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
  
---

## 🙋 Contact

[LinkedIn](https://linkedin.com/in/atheeth-naik-2679b5132) | Open for collaboration and questions.

---

<sup>Project by Atheeth Naik | Predictive Maintenance for Real-World Engineering</sup>
