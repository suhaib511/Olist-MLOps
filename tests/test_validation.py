from src.validation import validate_input


def test_valid_input():
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

    validate_input(data)

def test_missing_feature():
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
    
    }

    try:
        validate_input(data)
    except ValueError:
        return

    raise AssertionError("Expected ValueError for missing feature")

def test_none_value():
    data = {
        "purchase_year": 2018,
        "purchase_month": None,
        "purchase_dayofweek": 3,
        "approval_delay_hours": 10,
        "estimated_delivery_days": 25,
        "count_item": 1,
        "total_price": 100,
        "total_freight": 20,
        "count_sequential": 1,
        "sum_pyament_value": 120,
    }

    try:
        validate_input(data)
    except ValueError:
        return

    raise AssertionError("Expected ValueError for None value")