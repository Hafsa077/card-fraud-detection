"""Evaluation helpers: business cost, threshold choice and drift monitoring."""
import numpy as np


def alert_cost(y_true, scores, amounts, threshold: float, review_cost: float = 5.0) -> float:
    """Total cost of running the model at `threshold`.

    Every alert sends a transaction to manual review, costing `review_cost`.
    Every fraud the model misses costs its full transaction amount.
    """
    y_true = np.asarray(y_true)
    flagged = np.asarray(scores) >= threshold
    missed_fraud = (~flagged) & (y_true == 1)
    return float(flagged.sum() * review_cost + np.asarray(amounts)[missed_fraud].sum())


def best_threshold(y_true, scores, amounts, review_cost: float = 5.0, grid=None):
    """Pick the threshold with the lowest alert_cost. Returns (threshold, cost).

    Choose it on validation data, then report it once on the test set, so the test
    result is not tuned to the test set.
    """
    if grid is None:
        grid = np.round(np.linspace(0.01, 0.99, 99), 2)
    costs = [alert_cost(y_true, scores, amounts, t, review_cost) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between a reference sample and a newer one.

    Common rule of thumb: below 0.1 is stable, 0.1 to 0.25 is worth watching,
    above 0.25 means the input has shifted and the model should be reviewed.
    Bin edges come from the quantiles of the reference sample.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
