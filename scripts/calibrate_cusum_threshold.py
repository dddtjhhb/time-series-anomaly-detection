"""Estimate CUSUM false-alarm rates on series with no volatility change."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.cusum import upper_cusum


def main():
    n_simulations = 500
    n_observations = 500
    warmup = 100
    sigma = 0.01
    methods = ["mean_std", "median_mad"]
    thresholds = [8.0, 12.0, 16.0, 20.0, 24.0, 30.0, 40.0, 50.0]
    rows = []

    for seed in range(n_simulations):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0, sigma, n_observations)
        absolute_returns = pd.Series(np.abs(returns))

        for method in methods:
            for threshold in thresholds:
                monitored = upper_cusum(
                    absolute_returns,
                    warmup=warmup,
                    drift=0.5,
                    threshold=threshold,
                    baseline_method=method,
                )
                alarm_positions = np.flatnonzero(monitored["IsCUSUMAlarm"].to_numpy())
                first_alarm = int(alarm_positions[0]) if len(alarm_positions) else None
                rows.append(
                    {
                        "Seed": seed,
                        "BaselineMethod": method,
                        "Threshold": threshold,
                        "FalseAlarm": first_alarm is not None,
                        "FirstAlarmIndex": first_alarm,
                    }
                )

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["BaselineMethod", "Threshold"])
        .agg(
            FalseAlarms=("FalseAlarm", "sum"),
            FalseAlarmRate=("FalseAlarm", "mean"),
        )
        .reset_index()
    )
    summary["Simulations"] = n_simulations

    output = ROOT / "results" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "cusum_calibration_runs.csv", index=False)
    summary.to_csv(output / "cusum_calibration_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
