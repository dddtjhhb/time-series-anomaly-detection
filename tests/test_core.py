import unittest
import numpy as np
import pandas as pd
from src.data import clean_prices
from src.evaluation import classification_metrics
from src.methods import add_rolling_statistics, flag_anomalies
from src.synthetic import make_synthetic_series

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

if __name__ == "__main__": unittest.main()
