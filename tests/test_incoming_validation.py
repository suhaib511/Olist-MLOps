from src.data_validation import validate_incoming_data


def test_valid_incoming_data():
    data = {
        "purchase_year": 2018,
        "purchase_month": 7,
        "purchase_dayofweek": 3,
        "approval_delay_hours": 10.0,
        "estimated_delivery_days": 25.0,
        "count_item": 1.0,
        "total_price": 100.0,
        "total_freight": 20.0,
        "count_sequential": 1.0,
        "sum_pyament_value": 120.0,
    }

    validate_incoming_data(data)


def test_invalid_incoming_data():
    data = {
        "purchase_year": 2018,
        "purchase_month": 13,
        "purchase_dayofweek": 3,
        "approval_delay_hours": 10.0,
        "estimated_delivery_days": 25.0,
        "count_item": 1.0,
        "total_price": 100.0,
        "total_freight": 20.0,
        "count_sequential": 1.0,
        "sum_pyament_value": 120.0,
    }

    try:
        validate_incoming_data(data)
    except ValueError:
        return

    raise AssertionError("Expected ValueError for invalid incoming data")