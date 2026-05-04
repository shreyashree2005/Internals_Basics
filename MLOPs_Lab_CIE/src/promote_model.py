import mlflow
from mlflow.tracking import MlflowClient
import json
import os

client = MlflowClient()

MODEL_NAME = "commuhealth-outbreak-risk-score-predictor"

# Get all versions
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

# Sort versions
versions = sorted(versions, key=lambda x: int(x.version))

# Default assumption
champion_version = int(versions[0].version)
challenger_version = int(versions[-1].version)

# Compare MAE
run1 = client.get_run(versions[0].run_id)
run2 = client.get_run(versions[-1].run_id)

mae_v1 = run1.data.metrics.get("mae", 999)
mae_v2 = run2.data.metrics.get("mae", 999)

# Decide best model
if mae_v2 < mae_v1:
    champion_version = int(versions[-1].version)
    action = "promoted"
else:
    champion_version = int(versions[0].version)
    action = "kept"

# Set alias
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version=champion_version
)

# Output JSON (EXACT FORMAT)
output = {
    "registered_model_name": MODEL_NAME,
    "alias_name": "production",
    "champion_version": champion_version,
    "challenger_version": challenger_version,
    "action": action
}

os.makedirs("../results", exist_ok=True)

with open("../results/step4_s7.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed")