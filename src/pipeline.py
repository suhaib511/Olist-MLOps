import time
from src.loader import load_feature_list
from src.logging_config import setup_logging
from src.validation import validate_input
from src.data_validation import validate_incoming_data
from src.prediction import predict
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

MODEL_VERSION = config["mlflow"]["model_alias"]

logger = setup_logging()


def run_prediction(data: dict) -> dict:

    start_time = time.perf_counter()

    logger.info("Prediction request received | input=%s", data)

    try:
        validate_input(data)
        validate_incoming_data(data)

        feature_list = load_feature_list()

        result = predict(data, feature_list)

        latency = time.perf_counter() - start_time


        logger.info(
            "Prediction completed | prediction=%s | probability=%.4f | latency=%.4fs | model_version=%s",
            result["prediction"],
            result["probability"],
            latency,
            MODEL_VERSION,
        )



        return result

    except Exception:
        logger.exception("Prediction failed")
        raise