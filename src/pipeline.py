
import time
from pathlib import Path
import yaml

from src.loader import load_feature_list
from src.logging_config import setup_logging
from src.validation import validate_input
from src.data_validation import validate_incoming_data
from src.prediction import predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

MODEL_VERSION = config["mlflow"]["model_alias"]
logger = setup_logging()

monitoring_metrics = {
    "successful_predictions": 0,
    "failed_predictions": 0,
    "total_latency": 0.0,
}


def run_prediction(data: dict) -> dict:
    start_time = time.perf_counter()
    logger.info("Prediction request received | input=%s", data)

    try:
        validate_input(data)
        validate_incoming_data(data)

        feature_list = load_feature_list()
        result = predict(data, feature_list)

        latency = time.perf_counter() - start_time

        monitoring_metrics["successful_predictions"] += 1
        monitoring_metrics["total_latency"] += latency

        logger.info(
            "Prediction completed | prediction=%s | probability=%.4f | latency=%.4fs | model_version=%s",
            result["prediction"],
            result["probability"],
            latency,
            MODEL_VERSION,
        )

        return result

    except Exception:
        monitoring_metrics["failed_predictions"] += 1
        logger.exception("Prediction failed")
        raise


def get_monitoring_metrics() -> dict:
    successful = monitoring_metrics["successful_predictions"]
    failed = monitoring_metrics["failed_predictions"]

    return {
        "successful_predictions": successful,
        "failed_predictions": failed,
        "average_latency_seconds": (
            monitoring_metrics["total_latency"] / successful
            if successful else 0.0
        ),
        "model_version": MODEL_VERSION,
    }
