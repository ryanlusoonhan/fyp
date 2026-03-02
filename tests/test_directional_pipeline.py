import importlib
import pathlib
import tempfile
import unittest

import numpy as np
import torch

from src.dataset import StockDataset


class TestDirectionalPipeline(unittest.TestCase):
    @staticmethod
    def _write_minimal_rows(path: pathlib.Path, row_count: int) -> None:
        rows = ["Date,Close"]
        rows.extend([f"2026-01-{(idx % 28) + 1:02d},{100 + idx}" for idx in range(row_count)])
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_stock_dataset_emits_long_labels_for_classification(self):
        X = np.random.randn(8, 4, 3).astype(np.float32)
        y = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int64)
        ds = StockDataset(X, y)
        _, target = ds[0]
        self.assertEqual(target.dtype, torch.int64)

    def test_train_excludes_leaky_columns(self):
        train_mod = importlib.import_module("train")
        expected = {"Future_Close", "Future_Return", "Target_Class", "Recalculated_Target"}
        self.assertTrue(expected.issubset(train_mod.EXCLUDE_COLS))

    def test_weekly_strategy_simulation_uses_first_buy_signal_within_each_non_overlapping_period(self):
        weekly_mod = importlib.import_module("backtest_weekly")

        pred_class = np.array([0, 1, 1, 0, 1, 0], dtype=np.int64)
        aligned_future_ret = np.array([0.10, -0.05, 0.20, 0.03, -0.02, 0.01], dtype=np.float64)

        equity, bh_equity = weekly_mod.simulate_period_strategy(
            pred_class=pred_class,
            aligned_future_ret=aligned_future_ret,
            barrier_window=2,
            initial=100.0,
            cost=0.0,
        )

        # Non-overlapping with barrier_window=2 creates blocks [0,1], [2,3], [4,5].
        # AI executes at first BUY in each block using that row's return:
        # block1: BUY at idx1 -> -5%, block2: BUY at idx2 -> +20%, block3: BUY at idx4 -> -2%
        # start 100 -> 95 -> 114 -> 111.72
        self.assertAlmostEqual(equity[-1], 111.72, places=6)

        # Buy & hold over same non-overlapping periods: 100 * 1.10 * 1.20 * 0.98
        self.assertAlmostEqual(bh_equity[-1], 129.36, places=6)

    def test_requirements_pin_torch_compatible_sympy(self):
        req_path = pathlib.Path("requirements.txt")
        raw = req_path.read_bytes()

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-16le")

        lines = [line.strip().lstrip("\ufeff") for line in text.splitlines() if line.strip()]
        pins = {line.split("==", 1)[0]: line.split("==", 1)[1] for line in lines if "==" in line}

        self.assertEqual(pins.get("torch"), "2.6.0")
        self.assertEqual(pins.get("sympy"), "1.13.1")

    def test_threshold_optimizer_selects_best_f1_threshold(self):
        weekly_mod = importlib.import_module("backtest_weekly")

        probs_up = np.array([0.20, 0.40, 0.60, 0.80], dtype=np.float64)
        y_true = np.array([0, 0, 1, 1], dtype=np.int64)
        thresholds = np.array([0.30, 0.50, 0.70], dtype=np.float64)

        best_threshold, best_score, _ = weekly_mod.optimize_decision_threshold(
            probs_up=probs_up,
            y_true=y_true,
            objective="f1",
            thresholds=thresholds,
            aligned_future_ret=None,
            barrier_window=1,
            cost=0.0,
        )

        self.assertAlmostEqual(best_threshold, 0.50, places=8)
        self.assertAlmostEqual(best_score, 1.0, places=8)

    def test_threshold_optimizer_selects_best_return_threshold(self):
        weekly_mod = importlib.import_module("backtest_weekly")

        probs_up = np.array([0.49, 0.51, 0.52, 0.90], dtype=np.float64)
        y_true = np.array([0, 1, 1, 1], dtype=np.int64)
        aligned_future_ret = np.array([0.10, -0.20, 0.15, 0.02], dtype=np.float64)
        thresholds = np.array([0.50, 0.55], dtype=np.float64)

        best_threshold, best_score, _ = weekly_mod.optimize_decision_threshold(
            probs_up=probs_up,
            y_true=y_true,
            objective="return",
            thresholds=thresholds,
            aligned_future_ret=aligned_future_ret,
            barrier_window=1,
            cost=0.0,
        )

        self.assertAlmostEqual(best_threshold, 0.55, places=8)
        self.assertGreater(best_score, 0.0)

    def test_threshold_optimizer_selects_best_accuracy_threshold(self):
        weekly_mod = importlib.import_module("backtest_weekly")

        probs_up = np.array([0.15, 0.25, 0.45, 0.65, 0.85], dtype=np.float64)
        y_true = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        thresholds = np.array([0.30, 0.50, 0.70], dtype=np.float64)

        best_threshold, best_score, _ = weekly_mod.optimize_decision_threshold(
            probs_up=probs_up,
            y_true=y_true,
            objective="accuracy",
            thresholds=thresholds,
            aligned_future_ret=None,
            barrier_window=1,
            cost=0.0,
        )

        self.assertAlmostEqual(best_threshold, 0.50, places=8)
        self.assertAlmostEqual(best_score, 1.0, places=8)

    def test_walk_forward_slice_builder_creates_expected_windows(self):
        weekly_mod = importlib.import_module("backtest_weekly")

        windows = weekly_mod.build_walk_forward_slices(
            n_samples=250,
            window_size=100,
            step_size=75,
        )

        self.assertEqual(windows, [(0, 100), (75, 175), (150, 250)])

    def test_weekly_signal_helper_applies_threshold_consistently(self):
        weekly_inference = importlib.import_module("weekly_inference")

        pred_class, label = weekly_inference.classify_weekly_signal(prob_buy=0.61, threshold=0.55)
        self.assertEqual(pred_class, 1)
        self.assertEqual(label, "BUY")

        pred_class, label = weekly_inference.classify_weekly_signal(prob_buy=0.54, threshold=0.55)
        self.assertEqual(pred_class, 0)
        self.assertEqual(label, "NO_BUY")

    def test_train_weekly_prefers_primary_dataset_even_when_openbb_exists(self):
        train_weekly = importlib.import_module("train_weekly")

        with tempfile.TemporaryDirectory() as tmp:
            openbb_file = pathlib.Path(tmp) / "training_data_openbb.csv"
            legacy_file = pathlib.Path(tmp) / "training_data.csv"
            legacy_file.write_text("Date,Close\n2026-01-01,1\n", encoding="utf-8")
            openbb_file.write_text("Date,Close\n2026-01-01,2\n", encoding="utf-8")

            selected = train_weekly.resolve_training_data_path(
                openbb_path=str(openbb_file),
                fallback_path=str(legacy_file),
                min_rows=1,
            )
            self.assertEqual(selected, str(legacy_file))

    def test_weekly_inference_prefers_primary_dataset_even_if_config_points_to_openbb(self):
        weekly_inference = importlib.import_module("weekly_inference")

        original_openbb = weekly_inference.OPENBB_TRAINING_FILE
        original_processed = weekly_inference.PROCESSED_DATA_PATH

        with tempfile.TemporaryDirectory() as tmp:
            openbb_file = pathlib.Path(tmp) / "training_data_openbb.csv"
            primary_file = pathlib.Path(tmp) / "training_data.csv"
            self._write_minimal_rows(primary_file, 40)
            openbb_rows = ["Date,Close,VIX_Close"]
            openbb_rows.extend([f"2026-01-{(idx % 28) + 1:02d},{100 + idx},{20 + idx * 0.1:.2f}" for idx in range(40)])
            openbb_file.write_text("\n".join(openbb_rows) + "\n", encoding="utf-8")

            weekly_inference.OPENBB_TRAINING_FILE = str(openbb_file)
            weekly_inference.PROCESSED_DATA_PATH = f"{tmp}/"
            try:
                selected = weekly_inference.resolve_inference_data_path(
                    config_data_file=str(openbb_file),
                    required_columns=["Sentiment_MA7"],
                )
            finally:
                weekly_inference.OPENBB_TRAINING_FILE = original_openbb
                weekly_inference.PROCESSED_DATA_PATH = original_processed

            self.assertEqual(selected, str(primary_file))

    def test_scenario_prefers_primary_dataset_even_if_config_points_to_openbb(self):
        scenario_weekly = importlib.import_module("scenario_weekly")

        original_openbb = scenario_weekly.OPENBB_TRAINING_FILE
        original_processed = scenario_weekly.PROCESSED_DATA_PATH

        with tempfile.TemporaryDirectory() as tmp:
            openbb_file = pathlib.Path(tmp) / "training_data_openbb.csv"
            primary_file = pathlib.Path(tmp) / "training_data.csv"
            self._write_minimal_rows(openbb_file, 40)
            self._write_minimal_rows(primary_file, 40)

            scenario_weekly.OPENBB_TRAINING_FILE = str(openbb_file)
            scenario_weekly.PROCESSED_DATA_PATH = f"{tmp}/"
            try:
                selected = scenario_weekly.resolve_data_path(
                    config_data_file=str(openbb_file),
                    override_data_file=None,
                )
            finally:
                scenario_weekly.OPENBB_TRAINING_FILE = original_openbb
                scenario_weekly.PROCESSED_DATA_PATH = original_processed

            self.assertEqual(selected, str(primary_file))

    def test_weekly_inference_payload_remains_backward_compatible(self):
        weekly_inference = importlib.import_module("weekly_inference")

        payload = weekly_inference.build_signal_payload(
            as_of_date="2026-03-01",
            last_close=25000.0,
            threshold=0.5,
            objective="return",
            pred_class=1,
            label="BUY",
            probability_buy=0.62,
            model_version="best_model_weekly_binary.pth",
            data_status="fresh",
            last_refresh_at="2026-03-01T10:00:00Z",
            latest_market_date="2026-02-28",
        )

        required_keys = {
            "as_of_date",
            "last_close",
            "threshold",
            "objective",
            "pred_class",
            "label",
            "probability_buy",
            "probability_no_buy",
            "model_version",
        }
        self.assertTrue(required_keys.issubset(set(payload.keys())))
        self.assertEqual(payload["data_status"], "fresh")
        self.assertEqual(payload["latest_market_date"], "2026-02-28")


if __name__ == "__main__":
    unittest.main()
