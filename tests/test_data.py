import pandas as pd
import pytest

from fraud.data import time_split


def shuffled_frame(n=100):
    df = pd.DataFrame({"Time": range(n), "Class": [0] * n})
    return df.sample(frac=1, random_state=0)


def test_time_split_keeps_every_row_once():
    train, valid, test = time_split(shuffled_frame())
    assert len(train) + len(valid) + len(test) == 100
    assert set(train.Time) | set(valid.Time) | set(test.Time) == set(range(100))


def test_time_split_never_trains_on_the_future():
    train, valid, test = time_split(shuffled_frame())
    assert train.Time.max() < valid.Time.min()
    assert valid.Time.max() < test.Time.min()


@pytest.mark.parametrize("train_frac,valid_frac", [(0.8, 0.2), (0, 0.5), (0.6, 0)])
def test_time_split_rejects_bad_fractions(train_frac, valid_frac):
    with pytest.raises(ValueError):
        time_split(shuffled_frame(), train_frac, valid_frac)
