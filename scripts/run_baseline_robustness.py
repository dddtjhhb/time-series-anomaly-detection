"""Compare classical and robust warm-up estimates under point contamination."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.cusum import estimate_baseline


def main():
    rows = []
    methods = ["mean_std", "median_mad"]
    contamination_counts = [0, 1, 5, 10]

    for seed in range(100):
        rng = np.random.default_rng(seed)
        clean = pd.Series(np.abs(rng.normal(0, 0.01, 100)))
        for method in methods:
            clean_center, clean_scale = estimate_baseline(clean, method)
            for count in contamination_counts:
                contaminated = clean.copy()
                if count:
                    locations = rng.choice(100, count, replace=False)
                    contaminated.iloc[locations] = 0.10
                center, scale = estimate_baseline(contaminated, method)
                rows.append(
                    {
                        "Seed": seed,
                        "Method": method,
                        "ContaminatedPoints": count,
                        "CenterAbsoluteChange": abs(center - clean_center),
                        "ScaleAbsoluteChange": abs(scale - clean_scale),
                    }
                )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["Method", "ContaminatedPoints"])[
            ["CenterAbsoluteChange", "ScaleAbsoluteChange"]
        ]
        .mean()
        .reset_index()
    )
    output = ROOT / "results" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "baseline_contamination_runs.csv", index=False)
    summary.to_csv(output / "baseline_contamination_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
