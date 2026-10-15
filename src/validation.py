required_features = [
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


def validate_input(data: dict) -> None:

    if not isinstance(data, dict):
        raise TypeError("Input data must be a dictionary.")

    missing_features = [
        feature for feature in required_features
        if feature not in data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    imputable_features = [
        "approval_delay_hours",
        "count_sequential",
        "sum_pyament_value",
    ]

    for feature in required_features:
        value = data[feature]

        if value is None and feature not in imputable_features:
            raise ValueError(
                f"Feature '{feature}' cannot be None."
            )