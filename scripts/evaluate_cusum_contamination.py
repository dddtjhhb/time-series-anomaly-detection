"""Compare calibrated CUSUM baselines after warm-up contamination."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cusum import first_alarm_summary, upper_cusum
from src.synthetic import make_volatility_change_series


def main():
    n_simulations = 500
    evaluation_seeds = range(2000, 2000 + n_simulations)
    warmup = 100
    change_index = 500
    contamination_counts = [0, 1, 5, 10]
    contamination_value = 0.10
    calibrated_methods = {"mean_std": 12.0, "median_mad": 40.0}
    rows = []

    for seed in evaluation_seeds:
        data = make_volatility_change_series(
            n=1000,
            change_index=change_index,
            sigma_before=0.01,
            sigma_after=0.02,
            seed=seed,
        )
        location_rng = np.random.default_rng(seed + 100_000)
        ordered_locations = location_rng.permutation(warmup)

        for count in contamination_counts:
            observed = data["AbsReturn"].copy()
            if count:
                observed.iloc[ordered_locations[:count]] = contamination_value

            for method, threshold in calibrated_methods.items():
                monitored = upper_cusum(
                    observed,
                    warmup=warmup,
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
                        "Threshold": threshold,
                        "ContaminatedPoints": count,
                        "BaselineCenter": monitored["BaselineCenter"].iloc[0],
                        "BaselineScale": monitored["BaselineScale"].iloc[0],
                        **result,
                    }
                )

    runs = pd.DataFrame(rows)
    runs["Missed"] = ~(runs["Detected"] | runs["FalseAlarmBeforeChange"])
    summary = (
        runs.groupby(["BaselineMethod", "Threshold", "ContaminatedPoints"])
        .agg(
            MeanBaselineCenter=("BaselineCenter", "mean"),
            MeanBaselineScale=("BaselineScale", "mean"),
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
    runs.to_csv(output / "cusum_contamination_runs.csv", index=False)
    summary.to_csv(output / "cusum_contamination_summary.csv", index=False)

    figures = ROOT / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = {"mean_std": "Mean/SD", "median_mad": "Median/MAD"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method, group in summary.groupby("BaselineMethod"):
        label = labels[method]
        axes[0].plot(
            group["ContaminatedPoints"],
            group["DetectionRate"],
            marker="o",
            label=label,
        )
        axes[1].plot(
            group["ContaminatedPoints"],
            group["MedianDetectionDelay"],
            marker="o",
            label=label,
        )
    axes[0].set(
        xlabel="Contaminated warm-up observations",
        ylabel="Detection rate",
        ylim=(-0.05, 1.05),
    )
    axes[1].set(
        xlabel="Contaminated warm-up observations",
        ylabel="Median detection delay",
    )
    for axis in axes:
        axis.legend(frameon=False)
        axis.grid(alpha=0.25)
    fig.suptitle("CUSUM robustness under warm-up contamination")
    fig.tight_layout()
    fig.savefig(figures / "cusum_contamination.png", dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
