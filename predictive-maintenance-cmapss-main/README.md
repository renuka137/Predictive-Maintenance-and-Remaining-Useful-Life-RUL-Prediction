# Predictive Maintenance & Remaining Useful Life (RUL) Prediction

## 📌 Project Overview

Predictive maintenance helps identify potential equipment failures before they occur by analyzing historical sensor data.

This project focuses on **Remaining Useful Life (RUL) prediction of aircraft engines** using multi-cycle sensor data. Machine Learning and Deep Learning models are used to estimate how many operating cycles an engine can continue to operate before failure.

The project includes **data preprocessing, exploratory data analysis, RUL target engineering, Machine Learning and Deep Learning model development, model evaluation, and FastAPI deployment**.

---

## 🎯 Objectives

* Predict the **Remaining Useful Life (RUL)** of aircraft engines.
* Analyze sensor behavior across different operating cycles.
* Identify degradation patterns from engine sensor data.
* Compare **XGBoost** and **LSTM** regression models.
* Deploy the prediction model through a **FastAPI REST API**.
* Provide an interactive **Swagger UI** for testing predictions.

---

## 📊 Dataset

The dataset contains engine sensor measurements collected over multiple operating cycles.

### Dataset Features

| Feature Group            | Description                   |
| ------------------------ | ----------------------------- |
| `unit_id`                | Unique engine identifier      |
| `cycle`                  | Operating cycle of the engine |
| `op_setting_1`           | Operating condition 1         |
| `op_setting_2`           | Operating condition 2         |
| `op_setting_3`           | Operating condition 3         |
| `sensor_1` – `sensor_21` | Engine sensor measurements    |

The dataset contains **3 operating settings and 21 sensor measurements** for each engine operating cycle.

### RUL Calculation

The target variable is created using the engine's maximum operating cycle:

```text
RUL = Maximum Cycle of Engine - Current Cycle
```

For example:

```text
Maximum Cycle = 150
Current Cycle = 100

RUL = 150 - 100
    = 50 cycles
```

---

## 🔄 Project Workflow

```text
Raw Engine Sensor Data
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Selection & Scaling
        ↓
RUL Target Engineering
        ↓
Train/Test Split
        ↓
 ┌───────────────────────┐
 │   Machine Learning    │
 │      XGBoost          │
 └───────────────────────┘
        ↓
 ┌───────────────────────┐
 │   Deep Learning       │
 │       LSTM            │
 └───────────────────────┘
        ↓
Model Evaluation
        ↓
FastAPI Deployment
        ↓
Swagger UI Prediction
```

---

## 🤖 Models Used

### 1. XGBoost

XGBoost was used as a Machine Learning regression model to predict continuous RUL values from engine sensor and operating-condition features.

### 2. LSTM

Long Short-Term Memory (LSTM) was used because engine sensor data is sequential and changes over operating cycles.

LSTM can learn temporal patterns and degradation behavior from sequences of sensor measurements.

---

## 📈 Model Performance

The models were evaluated using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.

| Model   |     MAE |    RMSE |
| ------- | ------: | ------: |
| XGBoost | 11.8322 | 15.2509 |
| LSTM    | 10.6712 | 16.0870 |

### Evaluation Metrics

**MAE (Mean Absolute Error)**
Measures the average absolute difference between the actual and predicted RUL.

**RMSE (Root Mean Squared Error)**
Measures prediction error while giving higher weight to larger errors.

---

## 🚀 FastAPI Deployment

The trained model was integrated with **FastAPI** to create a REST API for RUL prediction.

### API Workflow

```text
User / Client
     ↓
FastAPI Endpoint
     ↓
Input Validation
     ↓
Preprocessing
     ↓
Trained Model
     ↓
RUL Prediction
     ↓
JSON Response
```

### Example Prediction Response

```json
{
    "predicted_rul": 124.1666488647461
}
```

This represents an estimated **RUL of approximately 124.17 operating cycles** for that particular input.

---

## 📖 Swagger UI

FastAPI automatically provides interactive API documentation through Swagger UI.

The API can be tested using:

```text
/docs
```

Swagger allows users to enter sensor values and operating conditions and receive an RUL prediction without building a separate frontend application.

---

## 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Deep Learning

* TensorFlow
* Keras
* LSTM

### Deployment

* FastAPI
* Uvicorn
* Swagger UI

### Development

* Jupyter Notebook
* VS Code
* Git & GitHub

---

## 📁 Project Structure

```text
Predictive-Maintenance-and-Remaining-Useful-Life-RUL-Prediction/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── RUL_Analysis_Model.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_xgboost.py
│   └── train_lstm.py
│
├── api/
│   └── main.py
│
├── models/
│   └── trained_models
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 💡 Business Application

Predictive maintenance can help engineering and maintenance teams:

* Monitor equipment health.
* Estimate remaining operating life.
* Identify potential degradation earlier.
* Reduce unexpected equipment failures.
* Support maintenance planning.
* Reduce unnecessary preventive maintenance.
* Improve equipment availability and reliability.

---

## 🔮 Future Scope

* Add real-time sensor data streaming.
* Build a dashboard using Power BI or Streamlit.
* Add more advanced time-series models such as GRU and Transformers.
* Implement real-time model monitoring.
* Add automated maintenance alerts.
* Improve RUL prediction using feature engineering and hyperparameter tuning.

---

## 👩‍💻 Author

**Renuka Patil**

Data Science / Data Analytics Enthusiast

---

## ⭐ Project Highlights

* Processed **21 sensor features** and **3 operating settings**.
* Engineered RUL targets using operating-cycle information.
* Compared **XGBoost and LSTM** models.
* Achieved **15.2509 RMSE with XGBoost**.
* Achieved **10.6712 MAE with LSTM**.
* Deployed the prediction model using **FastAPI**.
* Provided interactive testing through **Swagger UI**.
