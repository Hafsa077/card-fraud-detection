import numpy as np
import pandas as pd

from fraud.features import add_features, feature_columns


def make_rows(times, amounts):
    df = pd.DataFrame({"Time": times, "Amount": amounts, "Class": 0})
    for i in range(1, 29):
        df[f"V{i}"] = 0.0
    return df


def test_hour_wraps_every_24_hours():
    one_day = 24 * 3600
    out = add_features(make_rows([0, 3599, 3600, one_day + 2 * 3600], [1, 1, 1, 1]))
    assert out["hour"].tolist() == [0, 0, 1, 2]


def test_is_night_covers_midnight_to_six():
    out = add_features(make_rows([5 * 3600 + 59, 6 * 3600, 23 * 3600], [1, 1, 1]))
    assert out["is_night"].tolist() == [1, 0, 0]


def test_log_amount_handles_zero_and_negative():
    out = add_features(make_rows([0, 0, 0], [0.0, -5.0, np.e - 1]))
    assert out["log_amount"].tolist()[:2] == [0.0, 0.0]
    assert np.isclose(out["log_amount"].iloc[2], 1.0)


def test_add_features_does_not_modify_input():
    df = make_rows([0], [10])
    add_features(df)
    assert "hour" not in df.columns


def test_feature_columns_leave_out_raw_time_amount_and_label():
    cols = feature_columns(add_features(make_rows([0], [10])))
    assert len(cols) == 31
    assert not {"Time", "Amount", "Class"} & set(cols)
