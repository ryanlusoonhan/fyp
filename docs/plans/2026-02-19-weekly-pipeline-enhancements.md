# Weekly Pipeline Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement weekly-only inference, threshold tuning, and walk-forward evaluation for the active weekly model workflow.

**Architecture:** Extend `backtest_weekly.py` with threshold-search and walk-forward utilities and add a new `weekly_inference.py` entrypoint that loads weekly artifacts and emits the latest BUY/NO_BUY signal. Keep logic deterministic and test pure decision functions in unit tests.

**Tech Stack:** Python 3, PyTorch, NumPy, Pandas, scikit-learn, matplotlib, unittest.

### Task 1: Add failing tests for new weekly evaluation utilities

**Files:**
- Modify: `tests/test_directional_pipeline.py`

**Step 1: Write failing tests**
- Add tests for:
  - F1-based threshold selection.
  - Return-based threshold selection.
  - Walk-forward slice generation logic.
  - Weekly signal mapping helper.

**Step 2: Run tests to verify failures**
- Run: `python -m unittest discover -s tests -v`
- Expected: New tests fail due missing functions/behavior.

### Task 2: Implement threshold tuning and walk-forward evaluation

**Files:**
- Modify: `backtest_weekly.py`

**Step 1: Add minimal implementation**
- Add pure helpers:
  - threshold optimization function.
  - walk-forward slice generator.
  - rolling evaluation summary.
- Integrate helpers into CLI backtest flow.

**Step 2: Verify tests**
- Run: `python -m unittest discover -s tests -v`
- Expected: Newly added tests pass.

### Task 3: Implement weekly inference entrypoint

**Files:**
- Create: `weekly_inference.py`

**Step 1: Add latest signal workflow**
- Load weekly model config/scaler/checkpoint.
- Build latest sequence with engineered features.
- Optionally tune threshold from validation set.
- Print BUY/NO_BUY report with probability and threshold.

**Step 2: Add/adjust tests if needed**
- Keep tests focused on pure helpers where possible.

### Task 4: Verification and dependency hygiene

**Files:**
- Modify: `requirements.txt`

**Step 1: Ensure runtime dependency includes plotting package**
- Add `matplotlib` pin so weekly scripts run in fresh env.

**Step 2: End-to-end verification**
- Run:
  - `python -m unittest discover -s tests -v`
  - `python backtest_weekly.py --objective f1`
  - `python weekly_inference.py --objective f1`

