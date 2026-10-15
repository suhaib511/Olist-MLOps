
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_endpoint():
    response = client.get("/model")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "OlistLatePrediction"
    assert data["model_version"] == "champion"


def test_predict_endpoint():
    payload = {
        "purchase_year": 2018,
        "purchase_month": 7,
        "purchase_dayofweek": 5,
        "approval_delay_hours": 10.0,
        "estimated_delivery_days": 25.0,
        "count_item": 1.0,
        "total_price": 100.0,
        "total_freight": 20.0,
        "count_sequential": 1.0,
        "sum_pyament_value": 120.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model_version"] == "champion"


def test_predict_rejects_invalid_input():
    payload = {
        "purchase_year": 2018,
        "purchase_month": 13,
        "purchase_dayofweek": 5,
        "approval_delay_hours": 10.0,
        "estimated_delivery_days": 25.0,
        "count_item": 1.0,
        "total_price": 100.0,
        "total_freight": 20.0,
        "count_sequential": 1.0,
        "sum_pyament_value": 120.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_accepts_missing_imputable_value():
    payload = {
        "purchase_year": 2018,
        "purchase_month": 7,
        "purchase_dayofweek": 5,
        "approval_delay_hours": None,
        "estimated_delivery_days": 25.0,
        "count_item": 1.0,
        "total_price": 100.0,
        "total_freight": 20.0,
        "count_sequential": 1.0,
        "sum_pyament_value": 120.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model_version"] == "champion"


def test_batch_predict_endpoint():
    instance = {
        "purchase_year": 2018,
        "purchase_month": 7,
        "purchase_dayofweek": 5,
        "approval_delay_hours": 10.0,
        "estimated_delivery_days": 25.0,
        "count_item": 1.0,
        "total_price": 100.0,
        "total_freight": 20.0,
        "count_sequential": 1.0,
        "sum_pyament_value": 120.0,
    }

    payload = {
        "instances": [instance, instance],
    }

    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert len(data["predictions"]) == 2

    for result in data["predictions"]:
        assert result["prediction"] in [0, 1]
        assert 0.0 <= result["probability"] <= 1.0
        assert result["model_version"] == "champion"