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
