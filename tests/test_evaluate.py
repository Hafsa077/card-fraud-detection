import numpy as np

from fraud.evaluate import alert_cost, best_threshold, psi

Y = [1, 0, 1, 0]
SCORES = [0.9, 0.8, 0.1, 0.2]
AMOUNTS = [100.0, 50.0, 30.0, 20.0]


def test_alert_cost_counts_reviews_and_missed_fraud():
    # Two alerts at $5 each, plus the $30 fraud the model missed.
    assert alert_cost(Y, SCORES, AMOUNTS, threshold=0.5, review_cost=5) == 40.0


def test_best_threshold_picks_the_cheapest_option():
    threshold, cost = best_threshold(Y, SCORES, AMOUNTS, review_cost=5, grid=np.array([0.05, 0.5, 0.95]))
    # 0.05 flags everything: 4 reviews, nothing missed = $20.
    assert (threshold, cost) == (0.05, 20.0)


def test_psi_is_near_zero_for_the_same_distribution():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=10_000)
    assert psi(sample, sample) < 1e-9


def test_psi_flags_a_shifted_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 10_000), rng.normal(1, 1, 10_000)) > 0.25
