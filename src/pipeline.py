import time
from src.loader import load_feature_list
from src.logging_config import setup_logging
from src.validation import validate_input
from src.prediction import predict


logger = setup_logging()


def run_prediction(data: dict) -> dict:

    start_time = time.perf_counter()

    logger.info("Prediction request received | input=%s", data)

    try:
        validate_input(data)

        feature_list = load_feature_list()

        result = predict(data, feature_list)

        latency = time.perf_counter() - start_time


        logger.info(
            "Prediction completed | prediction=%s | probability=%.4f | latency=%.4fs | model_version=%s",
            result["prediction"],
            result["probability"],
            latency,
            "local",
        )



        return result

    except Exception:
        logger.exception("Prediction failed")
        raise