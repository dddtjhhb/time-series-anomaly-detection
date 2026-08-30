"""Compare a volatility baseline with a CUSUM-augmented linear model."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.forecasting import (
    add_volatility_forecast_features,
    fit_linear_regression,
    predict_linear,
    regression_metrics,
)


def main():
    data = pd.read_csv(ROOT / "data" / "processed" / "real_returns.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    split_date = pd.Timestamp("2022-01-01")
    target = "FutureVolatility5"
    model_features = {
        "Past volatility only": ["PastVolatility20"],
        "Past volatility + CUSUM": ["PastVolatility20", "AdaptiveCUSUMScore"],
    }
    rows = []
    prediction_frames = []

    for ticker, ticker_data in data.groupby("Ticker"):
        featured = add_volatility_forecast_features(
            ticker_data.sort_values("Date").reset_index(drop=True)
        )
        train = featured.loc[featured["Date"] < split_date]
        test = featured.loc[featured["Date"] >= split_date].dropna(
            subset=[target, "PastVolatility20", "AdaptiveCUSUMScore"]
        )

        ticker_predictions = test[["Date", "Ticker", target]].copy()
        for model_name, features in model_features.items():
            coefficients = fit_linear_regression(train, features, target)
            predicted = predict_linear(test, features, coefficients)
            metrics = regression_metrics(test[target], predicted)
            rows.append(
                {
                    "Ticker": ticker,
                    "Model": model_name,
                    "TrainEnd": "2021-12-31",
                    "TestStart": "2022-01-01",
                    "TestObservations": len(test),
                    **metrics,
                }
            )
            ticker_predictions[model_name] = predicted
        prediction_frames.append(ticker_predictions)

    results = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    baseline_rmse = results.loc[results["Model"] == "Past volatility only"].set_index("Ticker")["RMSE"]
    augmented_mask = results["Model"] == "Past volatility + CUSUM"
    results.loc[augmented_mask, "RMSEImprovementPercent"] = results.loc[
        augmented_mask, "Ticker"
    ].map(lambda ticker: 100.0 * (baseline_rmse[ticker] - results.loc[
        augmented_mask & (results["Ticker"] == ticker), "RMSE"
    ].iloc[0]) / baseline_rmse[ticker])

    tables = ROOT / "results" / "tables"
    figures = ROOT / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    results.to_csv(tables / "volatility_forecast_summary.csv", index=False)
    predictions.to_csv(tables / "volatility_forecast_predictions.csv", index=False)

    plot_data = results.pivot(index="Ticker", columns="Model", values="RMSE")
    axes = plot_data.plot(kind="bar", figsize=(8, 4), color=["#4C78A8", "#F58518"])
    axes.set(ylabel="Test RMSE (annualized volatility)", xlabel="ETF")
    axes.set_title("Out-of-sample 5-day volatility forecasting")
    axes.legend(frameon=False)
    axes.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures / "volatility_forecast_rmse.png", dpi=180)
    plt.close()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
