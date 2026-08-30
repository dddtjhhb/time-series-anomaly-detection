"""Synthetic return series with reproducible, known anomalies."""
import numpy as np
import pandas as pd

def make_synthetic_series(n=1500, n_anomalies=30, seed=42):
    if n < 100 or not 0 < n_anomalies < n - 100:
        raise ValueError("Choose n >= 100 and 0 < n_anomalies < n - 100")
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.01, n)
    baseline = np.zeros(n)
    for i in range(1, n):
        baseline[i] = 0.15 * baseline[i - 1] + noise[i]
    locations = rng.choice(np.arange(100, n), n_anomalies, replace=False)
    observed = baseline.copy()
    observed[locations] += rng.choice([-1.0, 1.0], n_anomalies) * rng.uniform(0.035, 0.065, n_anomalies)
    return pd.DataFrame({"Date": pd.date_range("2018-01-01", periods=n, freq="B"),
                         "Return": observed,
                         "IsInjected": np.isin(np.arange(n), locations)})


def make_volatility_change_series(
    n=1000, change_index=500, sigma_before=0.01, sigma_after=0.03, seed=42
):
    """Generate zero-mean returns with one known increase in volatility."""
    if not 100 <= change_index < n or sigma_before <= 0 or sigma_after <= 0:
        raise ValueError("choose a valid change index and positive volatilities")
    rng = np.random.default_rng(seed)
    returns = np.concatenate(
        [
            rng.normal(0, sigma_before, change_index),
            rng.normal(0, sigma_after, n - change_index),
        ]
    )
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=n, freq="B"),
            "Return": returns,
            "AbsReturn": np.abs(returns),
            "IsPostChange": np.arange(n) >= change_index,
        }
    )
