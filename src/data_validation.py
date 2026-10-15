import great_expectations as gx

context = gx.get_context(mode="file")

print(type(context).__name__)

data_source = context.data_sources.get("olist_pandas")

data_asset = data_source.get_asset("train_features")

batch_definition = data_asset.get_batch_definition(
    "train_features_batch"
)

expectation_suite = context.suites.get(
    "olist_train_expectations"
)

# Range

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnToExist(
        column="purchase_year"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="purchase_month",
        min_value=1,
        max_value=12
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="purchase_dayofweek",
        min_value=0,
        max_value=6
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="approval_delay_hours",
        min_value=0
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="estimated_delivery_days",
        min_value=0
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="count_item",
        min_value=1
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="total_price",
        min_value=0
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="total_freight",
        min_value=0
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="count_sequential",
        min_value=0
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="sum_pyament_value",
        min_value=0
    )
)

# Column Type

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeOfType(
        column="purchase_year",
        type_="int64"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeOfType(
        column="purchase_month",
        type_="int64"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeOfType(
        column="purchase_dayofweek",
        type_="int64"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeOfType(
        column="approval_delay_hours",
        type_="float64"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeOfType(
        column="estimated_delivery_days",
        type_="float64"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeOfType(
        column="count_item",
        type_="float64"
    )
)


expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeOfType(
        column="total_price",
        type_="float64"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeOfType(
        column="total_freight",
        type_="float64"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeOfType(
        column="count_sequential",
        type_="float64"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeOfType(
        column="sum_pyament_value",
        type_="float64"
    )
)

# Missing rates

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(
        column="purchase_year"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(
        column="purchase_month"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(
        column="purchase_dayofweek"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(
        column="estimated_delivery_days"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(
        column="count_item"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(
        column="total_price"
    )
)

expectation_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(
        column="total_freight"
    )
)














def validate_training_data() -> None:
    validation_definition = context.validation_definitions.get(
        "olist_train_validation"
    )

    try:
        checkpoint = context.checkpoints.get(
            "olist_train_checkpoint"
        )
    except Exception:
        checkpoint = context.checkpoints.add(
            gx.Checkpoint(
                name="olist_train_checkpoint",
                validation_definitions=[validation_definition]
            )
        )

    checkpoint_result = checkpoint.run()

    if not checkpoint_result.success:
        raise ValueError("Data validation failed. Request rejected.")

    print(checkpoint_result.describe())





import pandas as pd


def validate_incoming_data(data: dict) -> None:
    """Validate incoming API data with Great Expectations."""

    dataframe = pd.DataFrame([data])

    float_columns = [
        "approval_delay_hours",
        "estimated_delivery_days",
        "count_item",
        "total_price",
        "total_freight",
        "count_sequential",
        "sum_pyament_value",
    ]

    dataframe[float_columns] = dataframe[float_columns].astype("float64")

    validation_definition = context.validation_definitions.get(
        "incoming_request_validation"
    )

    result = validation_definition.run(
        batch_parameters={"dataframe": dataframe}
    )

    if not result.success:
        raise ValueError("Incoming data validation failed.")
