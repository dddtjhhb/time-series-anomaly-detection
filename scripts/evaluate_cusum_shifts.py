"""Evaluate a calibrated CUSUM on independent volatility-change simulations."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.cusum import first_alarm_summary, upper_cusum
from src.synthetic import make_volatility_change_series


def main():
    n_simulations = 500
    evaluation_seeds = range(1000, 1000 + n_simulations)
    sigma_before = 0.01
    sigma_after_values = [0.015, 0.02, 0.03]
    change_index = 500
    calibrated_methods = {"mean_std": 12.0, "median_mad": 40.0}
    rows = []

    for method, threshold in calibrated_methods.items():
        for sigma_after in sigma_after_values:
            for seed in evaluation_seeds:
                data = make_volatility_change_series(
                    n=1000,
                    change_index=change_index,
                    sigma_before=sigma_before,
                    sigma_after=sigma_after,
                    seed=seed,
                )
                monitored = upper_cusum(
                    data["AbsReturn"],
                    warmup=100,
                    drift=0.5,
                    threshold=threshold,
                    baseline_method=method,
                )
                result = first_alarm_summary(
                    monitored["IsCUSUMAlarm"], true_change_index=change_index
                )
                rows.append(
                    {
                        "Seed": seed,
                        "BaselineMethod": method,
                        "SigmaBefore": sigma_before,
                        "SigmaAfter": sigma_after,
                        "Threshold": threshold,
                        **result,
                    }
                )

    runs = pd.DataFrame(rows)
    runs["Missed"] = ~(runs["Detected"] | runs["FalseAlarmBeforeChange"])
    summary = (
        runs.groupby(
            ["BaselineMethod", "SigmaBefore", "SigmaAfter", "Threshold"]
        )
        .agg(
            FalseAlarmRate=("FalseAlarmBeforeChange", "mean"),
            DetectionRate=("Detected", "mean"),
            MissRate=("Missed", "mean"),
            MedianDetectionDelay=("DetectionDelay", "median"),
        )
        .reset_index()
    )
    summary["Simulations"] = n_simulations

    output = ROOT / "results" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "cusum_shift_runs.csv", index=False)
    summary.to_csv(output / "cusum_shift_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
