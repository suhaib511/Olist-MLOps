
from pathlib import Path
import joblib
import pickle
import mlflow
import mlflow.xgboost
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
import yaml
import os
from functools import lru_cache
import boto3
import tempfile
from urllib.parse import urlparse
from mlflow.tracking import MlflowClient

HttpArtifactRepository.thread_pool = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=1)


project_root = Path(__file__).resolve().parents[1]

config_path = project_root / "config" / "config.yaml"

with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    config["mlflow"]["tracking_uri"]
)

mlflow.set_tracking_uri(tracking_uri)

models_dir = project_root / "models"


def load_feature_list():
    path = models_dir / "feature_list.pkl"
    with open(path, "rb") as file:
        return pickle.load(file)


def load_imputer():
    path = models_dir / "imputer.pkl"
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_model():
    client = MlflowClient()
    version = client.get_model_version_by_alias(
        config["mlflow"]["registered_model_name"],
        config["mlflow"]["model_alias"]
    )

    bucket = "mlflow-artifacts"
    model_id = version.source.split("/")[-1]
    prefix = f"2/models/{model_id}/artifacts/"

    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    local_dir = tempfile.mkdtemp(prefix="olist_model_")

    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix
    )

    for obj in response.get("Contents", []):
        key = obj["Key"]
        filename = key[len(prefix):]
        if filename:
            s3.download_file(
                bucket,
                key,
                os.path.join(local_dir, filename)
            )

    return mlflow.xgboost.load_model(local_dir)
