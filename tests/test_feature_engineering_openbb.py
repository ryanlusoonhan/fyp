import unittest

import numpy as np
import pandas as pd

from src.utils import engineer_features_market_only


class TestFeatureEngineeringOpenBB(unittest.TestCase):
    def _build_input(self, rows: int = 40) -> pd.DataFrame:
        dates = pd.date_range("2025-01-01", periods=rows, freq="D")
        close = np.linspace(100.0, 140.0, rows)
        high = close + 2.0
        low = close - 2.0
        open_ = close - 0.5

        return pd.DataFrame(
            {
                "Date": dates,
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": np.linspace(1_000_000, 2_000_000, rows),
                "VIX_Close": np.linspace(15.0, 22.0, rows),
                "TNX_Close": np.linspace(3.0, 4.0, rows),
                "GSPC_Close": np.linspace(4_000.0, 4_500.0, rows),
                "USDHKD_Close": np.linspace(7.78, 7.82, rows),
                "USDCNY_Close": np.linspace(7.10, 7.30, rows),
                "HK_Breadth_Positive_1D": np.linspace(0.2, 0.8, rows),
                "HK_Breadth_Above_MA20": np.linspace(0.3, 0.9, rows),
                "HK_Breadth_Dispersion_5D": np.linspace(0.01, 0.04, rows),
            }
        )

    def test_adds_expected_market_features(self):
        df = engineer_features_market_only(self._build_input())

        expected_cols = {
            "HSI_Return_1D",
            "HSI_Return_5D",
            "HSI_Return_20D",
            "HSI_Volatility_10D",
            "HSI_Volatility_20D",
            "HSI_Range_Pct",
            "HSI_Gap_Pct",
            "VIX_Change_1D",
            "VIX_ZScore_20D",
            "TNX_Change_1D",
            "GSPC_Return_1D",
            "HSI_vs_GSPC_Return_Spread_1D",
            "USDHKD_Change_1D",
            "USDCNY_Change_1D",
            "HK_Breadth_Positive_1D",
            "HK_Breadth_Above_MA20",
            "HK_Breadth_Dispersion_5D",
            "Return_1D",
            "Volatility_10D",
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))

    def test_computes_past_only_returns_without_leakage(self):
        source = self._build_input()
        df = engineer_features_market_only(source)

        self.assertAlmostEqual(df["HSI_Return_1D"].iloc[0], 0.0, places=8)
        expected_5d = source["Close"].iloc[20] / source["Close"].iloc[15] - 1.0
        self.assertAlmostEqual(df["HSI_Return_5D"].iloc[20], expected_5d, places=8)

    def test_no_feature_nan_after_fill(self):
        df = engineer_features_market_only(self._build_input())
        feature_cols = [col for col in df.columns if col not in {"Date"}]
        self.assertFalse(df[feature_cols].isna().any().any())


if __name__ == "__main__":
    unittest.main()
