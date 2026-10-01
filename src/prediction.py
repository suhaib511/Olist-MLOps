from src.loader import load_model
from src.preprocessing import preprocess_input


def predict(data: dict, feature_list: list) -> dict:

    model = load_model()

    processed_data = preprocess_input(data, feature_list)

    prediction = model.predict(processed_data)[0]

    probability = model.predict_proba(processed_data)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
    }