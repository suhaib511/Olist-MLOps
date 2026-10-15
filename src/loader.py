from pathlib import Path
import joblib
import pickle
import mlflow
import yaml


project_root = Path(__file__).resolve().parents[1]

config_path = project_root / "config" / "config.yaml"

with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

models_dir = project_root / "models"


def load_feature_list():
    path = models_dir / "feature_list.pkl"
    with open(path,"rb") as file:
        return pickle.load(file)

def load_imputer():
    path = models_dir / "imputer.pkl"
    return joblib.load(path)

def load_model():
    model_uri = (
    f"models:/{config['mlflow']['registered_model_name']}"
    f"@{config['mlflow']['model_alias']}"
)
    return mlflow.xgboost.load_model(model_uri)
