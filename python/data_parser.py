from typing import Any

import numpy as np
import polars as pd
from polars import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from collections import Counter
from imblearn.over_sampling import SMOTE


def _preprocess(df: DataFrame) -> DataFrame:
    # Start by cleaning up, making everything lower case & converting data types
    df.columns = [col.lower() for col in df.columns]
    df.replace_column(
        df.get_column_index("class"), df.get_column("class").cast(pd.String)
    )
    # Scale amount by log
    df.insert_column(-1, pd.Series("amount_log", np.log(df["amount"] + 0.0001)))
    # Convert everything to float, otherwise we get errors with string - float operations
    df.replace_column(
        df.get_column_index("class"), df.get_column("class").cast(pd.Float64)
    )
    return df


def prepare_dataset(df: DataFrame) -> Any:
    df = _preprocess(df)
    x = df.drop(["class", "time"])
    y = df["class"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )

    print(x_train.shape)
    print(x_test.shape)

    scaler = RobustScaler()

    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(x_test)

    print("Original dataset shape %s" % Counter(y_train))

    smt = SMOTE(random_state=42)
    x_train_smt, y_train_smt = smt.fit_resample(x_train, y_train)

    print("Resampled dataset shape %s" % Counter(y_train_smt))

    return df, x_train_smt, y_train_smt, x_test, y_test
