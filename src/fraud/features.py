"""Feature engineering for card transactions."""
import numpy as np
import pandas as pd

SECONDS_PER_HOUR = 3600


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with engineered features added.

    hour        hour of day (0-23). Time counts seconds from the first transaction, and
                the dataset starts at midnight, so this wraps every 24 hours.
    is_night    1 for transactions between 00:00 and 05:59, when genuine spending is low.
    log_amount  log(1 + Amount). Amounts are heavily skewed; the log keeps a few very
                large payments from dominating the linear model.
    """
    out = df.copy()
    out["hour"] = ((out["Time"] // SECONDS_PER_HOUR) % 24).astype(int)
    out["is_night"] = (out["hour"] < 6).astype(int)
    out["log_amount"] = np.log1p(out["Amount"].clip(lower=0))
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: the 28 anonymised components plus the engineered features.

    Raw Time is left out on purpose. It only says how far into the two-day window a
    transaction happened, which will never repeat in production data.
    """
    pca = [c for c in df.columns if c.startswith("V")]
    return pca + ["hour", "is_night", "log_amount"]
