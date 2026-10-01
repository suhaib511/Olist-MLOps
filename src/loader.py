from pathlib import Path
import joblib
import pickle

project_root = Path(__file__).resolve().parents[1]

models_dir = project_root / "models"


def load_feature_list():
    path = models_dir / "feature_list.pkl"
    with open(path,"rb") as file:
        return pickle.load(file)

def load_imputer():
    path = models_dir / "imputer.pkl"
    return joblib.load(path)

def load_model():
    path = models_dir / "final_model.joblib"
    return joblib.load(path)
