"""Loading and splitting the credit card transactions dataset.

The data is the public "Credit Card Fraud Detection" dataset (European cardholders,
September 2013), published on OpenML as dataset 42397. It holds 284,807 transactions
over two days, of which 492 are fraud. (OpenML's older copy, 1597, marks Time as a row
id, so scikit-learn drops it; 42397 keeps it as a normal column.)

Columns:
    Time    seconds since the first transaction in the dataset
    V1-V28  anonymised features (PCA components; the original fields are confidential)
    Amount  transaction amount
    Class   1 = fraud, 0 = genuine
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

OPENML_ID = 42397
TARGET = "Class"


def load_transactions(data_home: str | Path = "../data") -> pd.DataFrame:
    """Download (first run only) and return the transactions, sorted by time."""
    # The "liac-arff" parser is used because scikit-learn's pandas ARFF parser fails on
    # pandas 3 when a category column holds numbers.
    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, data_home=str(data_home), parser="liac-arff")
    df = bunch.frame.copy()
    # Depending on the OpenML version the label is stored as 0/1 or as "False"/"True".
    labels = {"0": 0, "1": 1, "False": 0, "True": 1}
    df[TARGET] = df[TARGET].astype(str).map(labels)
    if df[TARGET].isna().any():
        raise ValueError(f"unexpected labels in {TARGET}: {sorted(bunch.frame[TARGET].astype(str).unique())}")
    df[TARGET] = df[TARGET].astype(int)
    df["Time"] = df["Time"].astype(float)
    df["Amount"] = df["Amount"].astype(float)
    return df.sort_values("Time", kind="mergesort").reset_index(drop=True)


def time_split(df: pd.DataFrame, train_frac: float = 0.6, valid_frac: float = 0.2):
    """Split by time, oldest rows first, so the model is always tested on later transactions.

    A random split would let the model see transactions from the future, which a real
    fraud system never can. Returns (train, valid, test).
    """
    if not 0 < train_frac < 1 or not 0 < valid_frac < 1 or train_frac + valid_frac >= 1:
        raise ValueError("fractions must be between 0 and 1 and leave room for a test set")
    ordered = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))
    return ordered.iloc[:train_end], ordered.iloc[train_end:valid_end], ordered.iloc[valid_end:]
