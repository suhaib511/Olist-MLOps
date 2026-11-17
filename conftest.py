import os

if os.name == "nt":
    os.environ.setdefault(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5001"
    )
