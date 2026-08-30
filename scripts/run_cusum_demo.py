"""Run the first online CUSUM experiment with one known volatility change."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.cusum import first_alarm_summary, upper_cusum
from src.synthetic import make_volatility_change_series


def main():
    change_index = 500
    data = make_volatility_change_series(change_index=change_index, seed=42)
    thresholds = [8.0, 12.0, 20.0, 30.0]
    rows = []

    for threshold in thresholds:
        monitored = upper_cusum(
            data["AbsReturn"], warmup=100, drift=0.5, threshold=threshold
        )
        rows.append(
            {
                "Threshold": threshold,
                **first_alarm_summary(monitored["IsCUSUMAlarm"], change_index),
            }
        )

    tables_dir = ROOT / "results" / "tables"
    figures_dir = ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(tables_dir / "cusum_thresholds.csv", index=False)

    chosen = upper_cusum(data["AbsReturn"], warmup=100, threshold=20.0)
    alarm_index = first_alarm_summary(chosen["IsCUSUMAlarm"], change_index)[
        "FirstAlarmIndex"
    ]

    display_end = 550
    shown = data.iloc[:display_end]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(shown.index, shown["AbsReturn"], color="#476A8A", linewidth=0.8)
    axes[0].axvline(change_index, color="#2A9D8F", linestyle="--", label="True change")
    axes[0].axvline(alarm_index, color="#D1495B", linestyle=":", label="First alarm")
    axes[0].set_ylabel("Absolute return")
    axes[0].legend(frameon=False)

    axes[1].plot(chosen.index[:display_end], chosen["CUSUMScore"].iloc[:display_end], color="#6C5B7B")
    axes[1].axhline(20.0, color="#D1495B", linestyle="--", label="Threshold = 20")
    axes[1].axvline(change_index, color="#2A9D8F", linestyle="--")
    axes[1].set(xlabel="Time index", ylabel="CUSUM score")
    axes[1].legend(frameon=False)
    fig.suptitle("Online CUSUM detection of a volatility increase")
    fig.tight_layout()
    fig.savefig(figures_dir / "cusum_demo.png", dpi=180)
    plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
