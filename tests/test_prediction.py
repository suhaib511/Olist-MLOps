from src.prediction import predict


def test_prediction_output():
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

    feature_list = [
        "purchase_year",
        "purchase_month",
        "purchase_dayofweek",
        "approval_delay_hours",
        "estimated_delivery_days",
        "count_item",
        "total_price",
        "total_freight",
        "count_sequential",
        "sum_pyament_value",
    ]

    result = predict(data, feature_list)

    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability"] <= 1