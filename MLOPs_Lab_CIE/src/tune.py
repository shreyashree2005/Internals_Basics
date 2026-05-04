import pandas as pd
import mlflow
import json
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("../data/training_data.csv")

X = df.drop("outbreak_risk_score", axis=1)
y = df["outbreak_risk_score"]

# Parameter grid (as per question)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 7, 15],
    "min_samples_split": [2, 4]
}

model = RandomForestRegressor(random_state=42)

mlflow.set_experiment("commuhealth-outbreak-risk-score")

with mlflow.start_run(run_name="tuning-commuhealth"):

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    grid.fit(X, y)

    best_params = grid.best_params_
    best_mae = -grid.best_score_

# Output JSON
output = {
    "search_type": "grid",
    "n_folds": 5,
    "total_trials": 18,
    "best_params": best_params,
    "best_mae": best_mae,
    "best_cv_mae": best_mae,
    "parent_run_name": "tuning-commuhealth"
}

os.makedirs("../results", exist_ok=True)

with open("../results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 2 completed")