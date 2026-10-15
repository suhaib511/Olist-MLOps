import sys
sys.stdout.reconfigure(encoding="utf-8")

import mlflow
import mlflow.xgboost
import joblib
import json
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

model = joblib.load(MODEL_PATH)

print(type(model).__name__)

with open(METRICS_PATH, "r", encoding="utf-8") as file:
    metrics = json.load(file)
print(metrics)

mlflow.set_experiment("Olist Late Prediction")

with mlflow.start_run(run_name="Olist Late Prediction - Final Model"):
    mlflow.log_params(model.get_params())

    validation_metrics = {
    f"validation_{key}": value
    for key, value in metrics["validation_before_final_training"].items()
}

test_metrics = {
    f"test_{key}": value
    for key, value in metrics["test_after_final_training"].items()
}

mlflow.xgboost.log_model(
    model,
    name="final_xgb",
    registered_model_name=config["mlflow"]["registered_model_name"]
)

mlflow.log_metrics(validation_metrics)
mlflow.log_metrics(test_metrics)


print("Tracking run completed.")