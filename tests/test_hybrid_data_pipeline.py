import unittest

import pandas as pd

from src.data.hybrid_data import build_hybrid_training_dataset, compute_openbb_news_sentiment


class TestHybridDataPipeline(unittest.TestCase):
    def test_compute_openbb_news_sentiment_scores_daily_headlines(self):
        news = pd.DataFrame(
            [
                {
                    "date": "2026-02-01T01:00:00Z",
                    "title": "HSI rallies on strong profit growth",
                    "summary": "bullish momentum and beat expectations",
                },
                {
                    "date": "2026-02-01T09:00:00Z",
                    "title": "Market gains as outlook improves",
                    "summary": "",
                },
                {
                    "date": "2026-02-02T02:00:00Z",
                    "title": "HSI drops after weak earnings miss",
                    "summary": "decline deepens on risk concerns",
                },
            ]
        )

        out = compute_openbb_news_sentiment(news)
        self.assertEqual(list(out.columns), ["Date", "OpenBB_News_Sentiment", "OpenBB_News_Article_Count"])
        self.assertEqual(len(out), 2)

        day1 = out[out["Date"] == pd.Timestamp("2026-02-01")]
        day2 = out[out["Date"] == pd.Timestamp("2026-02-02")]
        self.assertEqual(int(day1["OpenBB_News_Article_Count"].iloc[0]), 2)
        self.assertEqual(int(day2["OpenBB_News_Article_Count"].iloc[0]), 1)
        self.assertGreater(float(day1["OpenBB_News_Sentiment"].iloc[0]), float(day2["OpenBB_News_Sentiment"].iloc[0]))

    def test_build_hybrid_training_dataset_keeps_primary_rows_and_merges_secondary_features(self):
        primary = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03"]),
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000, 1100, 1200],
                "sentiment_score": [0.1, -0.2, 0.0],
                "Return": [0.0, 0.01, 0.01],
                "Target": [1, 0, 1],
            }
        )

        secondary = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-02-02", "2026-02-03"]),
                "VIX_Close": [20.0, 21.0],
                "TNX_Close": [4.0, 4.1],
                "GSPC_Close": [5000.0, 5010.0],
                "USDHKD_Close": [7.8, 7.8],
                "USDCNY_Close": [7.2, 7.21],
                "HK_Breadth_Positive_1D": [0.4, 0.6],
                "HK_Breadth_Above_MA20": [0.5, 0.7],
                "HK_Breadth_Dispersion_5D": [0.02, 0.03],
            }
        )

        news = pd.DataFrame(
            [
                {
                    "date": "2026-02-03T05:00:00Z",
                    "title": "HSI surges after strong guidance",
                    "summary": "gain and growth outlook",
                }
            ]
        )

        out = build_hybrid_training_dataset(primary, secondary, news)
        self.assertEqual(len(out), len(primary))
        self.assertEqual(list(out["Date"]), list(primary["Date"]))
        self.assertTrue({"VIX_Close", "TNX_Close", "OpenBB_News_Sentiment", "OpenBB_News_Article_Count"}.issubset(out.columns))
        self.assertAlmostEqual(float(out.loc[out["Date"] == pd.Timestamp("2026-02-01"), "VIX_Close"].iloc[0]), 0.0)
        self.assertGreaterEqual(float(out.loc[out["Date"] == pd.Timestamp("2026-02-03"), "OpenBB_News_Article_Count"].iloc[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
