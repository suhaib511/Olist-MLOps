from src.preprocessing import preprocess_input


def test_preprocess_valid_input():
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

    result = preprocess_input(data, feature_list)

    assert list(result.columns) == feature_list
    assert result.shape == (1, 10)

def test_preprocess_missing_value():
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

    result = preprocess_input(data, feature_list)

    assert result["approval_delay_hours"].isna().sum() == 0