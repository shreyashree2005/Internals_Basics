import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os

# Load data
df = pd.read_csv("../data/training_data.csv")

# Split data
X = df.drop("outbreak_risk_score", axis=1)
y = df["outbreak_risk_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow experiment
mlflow.set_experiment("commuhealth-outbreak-risk-score")

models = {
    "Ridge": Ridge(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

results = []
best_model = None
best_mae = float("inf")

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))   # ✅ FIXED

        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("team", "ml_engineering")

        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse
        })

        if mae < best_mae:
            best_mae = mae
            best_model = name

output = {
    "experiment_name": "commuhealth-outbreak-risk-score",
    "models": results,
    "best_model": best_model,
    "best_metric_name": "mae",
    "best_metric_value": best_mae
}

os.makedirs("../results", exist_ok=True)

with open("../results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed")