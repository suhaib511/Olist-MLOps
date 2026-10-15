from src.pipeline import run_prediction
import pytest


def test_run_prediction():
    data = {
        "purchase_year": 2018,
        "purchase_month": 7,
        "purchase_dayofweek": 3,
        "approval_delay_hours": 10,
        "estimated_delivery_days": 25,
        "count_item": 1,
        "total_price": 100,
        "total_freight": 20,
        "count_sequential": 1,
        "sum_pyament_value": 120,
    }

    result = run_prediction(data)

    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability"] <= 1

def test_run_prediction_with_missing_value():
    data = {
        "purchase_year": 2018,
        "purchase_month": 7,
        "purchase_dayofweek": 3,
        "approval_delay_hours": None,
        "estimated_delivery_days": 25,
        "count_item": 1,
        "total_price": 100,
        "total_freight": 20,
        "count_sequential": 1,
        "sum_pyament_value": 120,
    }

    result = run_prediction(data)

    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability"] <= 1


@pytest.mark.parametrize(
    "missing_feature",
    [
        "approval_delay_hours",
        "count_sequential",
        "sum_pyament_value",
    ],
)
def test_run_prediction_with_missing_imputed_feature(missing_feature):
    data = {
        "purchase_year": 2018,
        "purchase_month": 7,
        "purchase_dayofweek": 3,
        "approval_delay_hours": 10,
        "estimated_delivery_days": 25,
        "count_item": 1,
        "total_price": 100,
        "total_freight": 20,
        "count_sequential": 1,
        "sum_pyament_value": 120,
    }

    data[missing_feature] = None

    result = run_prediction(data)

    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability"] <= 1