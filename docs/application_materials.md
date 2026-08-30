# Application Materials

## Résumé bullet points

- Built a reproducible Python pipeline to clean 5,283 daily observations across SPY, QQQ, and TLT and evaluate rolling z-score anomaly detection over 18 window/threshold/asset combinations.
- Designed a seeded synthetic time-series experiment with 30 known injected shocks; implemented precision, recall, F1, and false-positive/false-negative analysis, achieving 0.783 F1 for the best pre-specified configuration.
- Prevented temporal information leakage with past-only rolling statistics; added automated tests, cached data ingestion, parameterized configuration, publication-ready plots, and documented one-command reproduction.
- Implemented and Monte Carlo-calibrated an online CUSUM volatility-change detector; compared classical mean/SD and robust median/MAD baselines at matched false-alarm rates under controlled warm-up contamination.
- Tested whether an adaptive CUSUM score improved 5-day volatility forecasts on a 2022–2024 holdout set; documented the negative result when the augmented linear model underperformed the recent-volatility baseline across SPY, QQQ, and TLT.

Use two or three bullets if space is tight. Keep “best pre-specified
configuration” and “negative result” because both accurately describe the
limited experiments without overstating their conclusions.

## 2–3 minute English pitch to a professor

Hello Professor [Name]. I’m a junior studying Statistics and Computer Science at UIUC, and I recently completed a small independent project on anomaly detection in time-series data. My goal was not to build a trading strategy or use an advanced model that I could not explain. Instead, I wanted to practice the full research workflow with a simple and interpretable statistical baseline.

I used daily adjusted closing prices for SPY, QQQ, and TLT and converted them to daily returns. For each day, I compared the return with the mean and standard deviation of either the previous 20 or 60 trading days. I then used rolling z-score thresholds of 2, 2.5, and 3 to flag unusual observations. One detail I paid attention to was shifting the rolling statistics by one day, so the current return did not influence the baseline used to evaluate itself.

The real data let me study how sensitive the number of flags was to the window and threshold, but it did not provide true anomaly labels. To evaluate accuracy more rigorously, I created a reproducible synthetic return series and injected 30 known shocks. I measured precision, recall, and F1 for all six parameter combinations. The 60-day window with a threshold of 2.5 had the best F1 in my small grid, with 27 true positives, 12 false positives, and 3 missed anomalies. More importantly, I could clearly see the tradeoff: stricter thresholds reduced false positives but missed more injected anomalies.

I also documented the pipeline, fixed the random seed, cached the raw data, and wrote tests for cleaning, rolling calculations, threshold behavior, metrics, and reproducibility. The main limitation is that the simulation is simplified and one random seed is not enough for a general conclusion. My next step would be repeated simulations with uncertainty intervals and a comparison with a robust median-based method.

This project helped me connect statistical reasoning with careful implementation, and I’m looking for an undergraduate research setting where I can strengthen those skills on a real scientific problem. I would be glad to show you the repository and discuss how this kind of workflow could contribute to your group.

## Likely follow-up questions

**Why returns instead of prices?** Prices often trend and are not directly comparable over time. Returns represent day-to-day relative changes and are closer to a stable baseline, although their variance still changes.

**What is a false positive here?** In synthetic data, it is a normal point flagged as anomalous. In real data we cannot use that label because ground truth is unknown.

**Why use F1?** It summarizes precision and recall with their harmonic mean. It is useful when both false alarms and misses matter, but it is not the only valid choice.

**Is 60 days and 2.5 optimal?** Only within this small grid and this simulation. The project does not claim universal optimality.
