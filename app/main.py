from fastapi import FastAPI, HTTPException
import yaml
from pathlib import Path
from src.pipeline import run_prediction
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
)

app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_info():
    return {
        "model_name": config["mlflow"]["registered_model_name"],
        "model_version": config["mlflow"]["model_alias"],
    }



@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    data = request.model_dump()

    try:
        result = run_prediction(data)
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    return {
        "prediction": result["prediction"],
        "probability": result["probability"],
        "model_version": config["mlflow"]["model_alias"],
    }



@app.post("/predict/batch")
def predict_batch_endpoint(request: BatchPredictionRequest):
    results = []

    for instance in request.instances:
        data = instance.model_dump()

        try:
            result = run_prediction(data)
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail=str(error),
            ) from error

        results.append({
            "prediction": result["prediction"],
            "probability": result["probability"],
            "model_version": config["mlflow"]["model_alias"],
        })

    return {"predictions": results}