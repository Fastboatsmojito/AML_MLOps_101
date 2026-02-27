"""
Data cleaning and feature engineering utilities for the Contoso service orders
(OS) classification use case.

Target: predict RepairType (Overhaul vs Preventive) from
structured equipment maintenance records.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


CATEGORICAL_COLS = ["EquipmentModel", "JobCode", "ServiceCenter"]
NUMERICAL_COLS = ["QtyOrdered", "month", "quarter", "day_of_week"]
TARGET_COL = "RepairType"
LABEL_MAP = {"Overhaul": 1, "Preventive": 0}


def load_and_clean_os(filepath: str) -> pd.DataFrame:
    """Load the service orders dataset, drop nulls, and engineer features."""
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    required = ["OrderID", TARGET_COL]
    df = df.dropna(subset=required).reset_index(drop=True)

    df["OrderRequestDate"] = pd.to_numeric(df["OrderRequestDate"], errors="coerce")
    date_series = pd.to_datetime(df["OrderRequestDate"].astype("Int64").astype(str),
                                 format="%Y%m%d", errors="coerce")
    df["month"] = date_series.dt.month.fillna(0).astype(int)
    df["quarter"] = date_series.dt.quarter.fillna(0).astype(int)
    df["day_of_week"] = date_series.dt.dayofweek.fillna(0).astype(int)

    df["QtyOrdered"] = pd.to_numeric(df["QtyOrdered"], errors="coerce").fillna(0)
    df["ServiceCenter"] = df["ServiceCenter"].astype(str)

    df["label"] = df[TARGET_COL].map(LABEL_MAP)
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    return df


def build_preprocessor(
    categorical_cols: list = None,
    numerical_cols: list = None,
    max_categories: int = 20,
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for tabular features."""
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  max_categories=max_categories,
                                  sparse_output=False),
             categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ],
        remainder="drop",
    )


def prepare_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    max_categories: int = 20,
):
    """End-to-end: split, build preprocessor, fit/transform.

    Returns X_train, X_test, y_train, y_test, preprocessor, train_df, test_df.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"],
    )

    preprocessor = build_preprocessor(max_categories=max_categories)

    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    return X_train, X_test, y_train, y_test, preprocessor, train_df, test_df
