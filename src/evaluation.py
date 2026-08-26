"""Evaluation metrics and experiment grids."""
import pandas as pd
from .methods import add_rolling_statistics, flag_anomalies

def classification_metrics(actual: pd.Series, predicted: pd.Series):
    actual, predicted = actual.astype(bool), predicted.astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP":tp,"FP":fp,"FN":fn,"TN":tn,"Precision":precision,"Recall":recall,"F1":f1}

def evaluate_grid(frame, windows, thresholds):
    rows = []
    for window in windows:
        scored = add_rolling_statistics(frame, window)
        for threshold in thresholds:
            flagged = flag_anomalies(scored, threshold)
            rows.append({"Window":window,"Threshold":threshold,
                         **classification_metrics(flagged["IsInjected"], flagged["IsAnomaly"])})
    return pd.DataFrame(rows)
