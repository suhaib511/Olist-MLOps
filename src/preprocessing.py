import pandas as pd

from src.loader import load_imputer


def preprocess_input(data: dict, feature_list: list) -> pd.DataFrame:

    input_df = pd.DataFrame([data])

    input_df = input_df[feature_list]

    imputer = load_imputer()

    imputer_features = imputer.feature_names_in_

    input_df[imputer_features] = imputer.transform(
        input_df[imputer_features]
    )

    return input_df