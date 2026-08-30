import unittest
import numpy as np
import pandas as pd
from src.data import clean_prices
from src.cusum import estimate_baseline, first_alarm_summary, upper_cusum
from src.evaluation import classification_metrics
from src.methods import add_rolling_statistics, flag_anomalies
from src.synthetic import make_synthetic_series, make_volatility_change_series

class TestCore(unittest.TestCase):
    def test_cleaning(self):
        raw=pd.DataFrame({"Date":["2024-01-02","2024-01-01","2024-01-01","bad"],"Close":[102,100,101,50]})
        result=clean_prices(raw,"spy")
        self.assertEqual(result["Close"].tolist(),[101,102])
        self.assertEqual(result["Ticker"].unique().tolist(),["SPY"])
    def test_past_only_rolling(self):
        result=add_rolling_statistics(pd.DataFrame({"Return":[1.,2.,3.,100.]}),3)
        self.assertAlmostEqual(result.loc[3,"RollingMean"],2.)
        self.assertAlmostEqual(result.loc[3,"RollingStd"],1.)
        self.assertAlmostEqual(result.loc[3,"ZScore"],98.)
    def test_strict_threshold(self):
        result=flag_anomalies(pd.DataFrame({"ZScore":[-3.,-2.1,2.,2.1,np.nan]}),2.)
        self.assertEqual(result["IsAnomaly"].tolist(),[True,True,False,True,False])
    def test_metrics(self):
        result=classification_metrics(pd.Series([1,1,0,0]),pd.Series([1,0,1,0]))
        self.assertEqual((result["TP"],result["FP"],result["FN"],result["TN"]),(1,1,1,1))
        self.assertEqual((result["Precision"],result["Recall"]),(0.5,0.5))
    def test_synthetic_reproducible(self):
        left=make_synthetic_series(200,5,7); right=make_synthetic_series(200,5,7)
        pd.testing.assert_frame_equal(left,right)
        self.assertEqual(int(left["IsInjected"].sum()),5)

    def test_cusum_is_online(self):
        early = pd.Series(([0.0, 1.0] * 20) + [0.5] * 10 + [1.0] * 10)
        changed_future = early.copy()
        changed_future.iloc[50:] = 100.0
        left = upper_cusum(early, warmup=20)
        right = upper_cusum(changed_future, warmup=20)
        pd.testing.assert_series_equal(left.loc[:49, "CUSUMScore"], right.loc[:49, "CUSUMScore"])

    def test_cusum_detects_volatility_increase(self):
        data = make_volatility_change_series(seed=42)
        monitored = upper_cusum(data["AbsReturn"], warmup=100, threshold=20.0)
        summary = first_alarm_summary(monitored["IsCUSUMAlarm"], true_change_index=500)
        self.assertFalse(summary["FalseAlarmBeforeChange"])
        self.assertTrue(summary["Detected"])
        self.assertGreaterEqual(summary["DetectionDelay"], 0)

    def test_robust_baseline_resists_one_extreme_value(self):
        clean = pd.Series(np.linspace(0.001, 0.02, 100))
        contaminated = clean.copy()
        contaminated.iloc[0] = 1.0
        classical_clean = estimate_baseline(clean, "mean_std")
        classical_dirty = estimate_baseline(contaminated, "mean_std")
        robust_clean = estimate_baseline(clean, "median_mad")
        robust_dirty = estimate_baseline(contaminated, "median_mad")
        self.assertGreater(
            abs(classical_dirty[0] - classical_clean[0]),
            abs(robust_dirty[0] - robust_clean[0]),
        )

if __name__ == "__main__": unittest.main()
