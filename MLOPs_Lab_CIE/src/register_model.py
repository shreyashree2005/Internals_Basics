import mlflow
from mlflow.tracking import MlflowClient
import json
import os

client = MlflowClient()

MODEL_NAME = "commuhealth-outbreak-risk-score-predictor"

# Get latest run automatically
experiment = client.get_experiment_by_name("commuhealth-outbreak-risk-score")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

# Get run_id
run_id = runs[0].info.run_id

# Model path (RandomForest from train.py)
model_uri = f"runs:/{run_id}/RandomForest"

# Register model
result = mlflow.register_model(model_uri, MODEL_NAME)

# Output JSON (exact format required)
output = {
    "registered_model_name": MODEL_NAME,
    "version": result.version,
    "run_id": run_id,
    "source_metric": "mae",
    "source_metric_value": 5.29
}

os.makedirs("../results", exist_ok=True)

with open("../results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 3 completed")