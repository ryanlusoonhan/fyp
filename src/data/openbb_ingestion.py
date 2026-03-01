from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    OPENBB_BATCH_START_DATE,
    OPENBB_CURRENCY_SYMBOLS,
    OPENBB_EQUITY_SYMBOLS,
    OPENBB_INDEX_SYMBOLS,
    OPENBB_MANIFEST_FILE,
    OPENBB_NEWS_FILE,
    OPENBB_PROVIDER,
    OPENBB_REFRESH_STATUS_FILE,
    OPENBB_TRAINING_FILE,
    OPENBB_CURRENCY_HISTORY_FILE,
    OPENBB_EQUITY_HISTORY_FILE,
    OPENBB_INDEX_HISTORY_FILE,
)
from src.data.openbb_client import OpenBBClient


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _to_date_string(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _normalize_symbol(value: str) -> str:
    return str(value).replace("=X", "")


def _compute_breadth_features(equity_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df.empty:
        return pd.DataFrame(columns=["date", "HK_Breadth_Positive_1D", "HK_Breadth_Above_MA20", "HK_Breadth_Dispersion_5D"])

    work = equity_df.copy()
    work["symbol_norm"] = work["symbol"].map(_normalize_symbol)
    work = work.sort_values(["symbol_norm", "date"]).reset_index(drop=True)

    grouped = work.groupby("symbol_norm", group_keys=False)
    work["eq_return_1d"] = grouped["close"].pct_change()
    work["eq_ma20"] = grouped["close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    work["eq_ret_5d"] = grouped["close"].pct_change(5)
    work["eq_above_ma20"] = (work["close"] > work["eq_ma20"]).astype(float)
    work["eq_positive_1d"] = (work["eq_return_1d"] > 0).astype(float)

    agg = (
        work.groupby("date", as_index=False)
        .agg(
            HK_Breadth_Positive_1D=("eq_positive_1d", "mean"),
            HK_Breadth_Above_MA20=("eq_above_ma20", "mean"),
            HK_Breadth_Dispersion_5D=("eq_ret_5d", "std"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    return agg


def _prepare_index_features(index_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = index_df.copy()
    work["symbol_norm"] = work["symbol"].map(_normalize_symbol)
    hsi = (
        work[work["symbol_norm"] == "^HSI"][["date", "open", "high", "low", "close", "volume"]]
        .rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        .copy()
    )
    if hsi.empty:
        raise ValueError("HSI base series is empty after OpenBB fetch.")

    pivot = work.pivot_table(index="date", columns="symbol_norm", values="close", aggfunc="last")
    rename_map = {
        "^VIX": "VIX_Close",
        "^TNX": "TNX_Close",
        "^GSPC": "GSPC_Close",
    }
    exog = pd.DataFrame(index=pivot.index)
    for symbol_norm, col_name in rename_map.items():
        exog[col_name] = pivot[symbol_norm] if symbol_norm in pivot.columns else pd.NA
    exog = exog.reset_index().rename(columns={"date": "Date"})
    return hsi, exog


def _prepare_currency_features(currency_df: pd.DataFrame) -> pd.DataFrame:
    if currency_df.empty:
        return pd.DataFrame(columns=["Date", "USDHKD_Close", "USDCNY_Close"])
    work = currency_df.copy()
    work["symbol_norm"] = work["symbol"].map(_normalize_symbol)
    pivot = work.pivot_table(index="date", columns="symbol_norm", values="close", aggfunc="last")
    out = pd.DataFrame(index=pivot.index)
    for symbol in ["USDHKD", "USDCNY"]:
        col = f"{symbol}_Close"
        out[col] = pivot[symbol] if symbol in pivot.columns else pd.NA
    return out.reset_index().rename(columns={"date": "Date"})


def build_training_dataset(
    index_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    currency_df: pd.DataFrame,
) -> pd.DataFrame:
    base_df, index_exog = _prepare_index_features(index_df)
    breadth_df = _compute_breadth_features(equity_df).rename(columns={"date": "Date"})
    currency_exog = _prepare_currency_features(currency_df)

    merged = base_df.merge(index_exog, on="Date", how="left")
    merged = merged.merge(currency_exog, on="Date", how="left")
    merged = merged.merge(breadth_df, on="Date", how="left")

    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce").dt.tz_localize(None)
    merged = merged.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    base_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    exog_cols = [col for col in merged.columns if col not in base_cols]
    merged[exog_cols] = merged[exog_cols].ffill().fillna(0.0)
    merged["Return"] = merged["Close"].pct_change().fillna(0.0)
    merged["Target"] = (merged["Close"].shift(-1) > merged["Close"]).astype(int)
    merged["Target"] = merged["Target"].fillna(0).astype(int)
    return merged


def resolve_refresh_window(
    mode: str,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = 180,
) -> tuple[str, str]:
    today = date.today()
    end = _to_date_string(end_date or today)
    if mode == "live":
        start = _to_date_string(start_date or (today - timedelta(days=max(1, int(lookback_days)))))
    else:
        start = _to_date_string(start_date or OPENBB_BATCH_START_DATE)
    return start, end


def load_refresh_status(status_file: str = OPENBB_REFRESH_STATUS_FILE) -> dict[str, Any]:
    if not os.path.exists(status_file):
        return {}
    with open(status_file, "r", encoding="utf-8") as file:
        return json.load(file)


def run_openbb_refresh(
    mode: str = "batch",
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = 180,
    write_training: bool = True,
    provider: str = OPENBB_PROVIDER,
    client: OpenBBClient | None = None,
) -> dict[str, Any]:
    if mode not in {"batch", "live"}:
        raise ValueError("mode must be one of {'batch', 'live'}.")

    start, end = resolve_refresh_window(mode=mode, start_date=start_date, end_date=end_date, lookback_days=lookback_days)
    openbb_client = client or OpenBBClient(provider=provider)

    index_df = openbb_client.fetch_index_history(symbols=OPENBB_INDEX_SYMBOLS, start_date=start, end_date=end)
    equity_df = openbb_client.fetch_equity_history(symbols=OPENBB_EQUITY_SYMBOLS, start_date=start, end_date=end)
    currency_df = openbb_client.fetch_currency_history(symbols=OPENBB_CURRENCY_SYMBOLS, start_date=start, end_date=end)

    try:
        news_df = openbb_client.fetch_recent_news(symbols=["^HSI", *OPENBB_EQUITY_SYMBOLS], limit=200)
    except Exception:
        news_df = pd.DataFrame(columns=["date", "title", "url", "source", "symbol", "summary", "text", "id"])

    dataset = build_training_dataset(index_df=index_df, equity_df=equity_df, currency_df=currency_df)

    for path in [
        OPENBB_INDEX_HISTORY_FILE,
        OPENBB_EQUITY_HISTORY_FILE,
        OPENBB_CURRENCY_HISTORY_FILE,
        OPENBB_NEWS_FILE,
        OPENBB_MANIFEST_FILE,
        OPENBB_REFRESH_STATUS_FILE,
        OPENBB_TRAINING_FILE,
    ]:
        _ensure_parent(path)

    index_df.to_csv(OPENBB_INDEX_HISTORY_FILE, index=False)
    equity_df.to_csv(OPENBB_EQUITY_HISTORY_FILE, index=False)
    currency_df.to_csv(OPENBB_CURRENCY_HISTORY_FILE, index=False)
    news_df.to_csv(OPENBB_NEWS_FILE, index=False)

    if write_training:
        dataset.to_csv(OPENBB_TRAINING_FILE, index=False)

    refresh_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    latest_market_date = dataset["Date"].max().date().isoformat() if len(dataset) else None
    if not latest_market_date:
        data_status = "empty"
    else:
        age_days = (date.today() - date.fromisoformat(latest_market_date)).days
        data_status = "stale" if age_days > 5 else "fresh"
    status = {
        "provider": provider,
        "mode": mode,
        "start_date": start,
        "end_date": end,
        "rows_index_history": int(len(index_df)),
        "rows_equity_history": int(len(equity_df)),
        "rows_currency_history": int(len(currency_df)),
        "rows_news_recent": int(len(news_df)),
        "rows_training": int(len(dataset)),
        "last_refresh_at": refresh_at,
        "latest_market_date": latest_market_date,
        "data_status": data_status,
        "training_file": OPENBB_TRAINING_FILE if write_training else None,
    }

    manifest = {
        "refresh": status,
        "index_symbols": OPENBB_INDEX_SYMBOLS,
        "equity_symbols": OPENBB_EQUITY_SYMBOLS,
        "currency_symbols": OPENBB_CURRENCY_SYMBOLS,
        "provider": provider,
    }

    with open(OPENBB_MANIFEST_FILE, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    with open(OPENBB_REFRESH_STATUS_FILE, "w", encoding="utf-8") as file:
        json.dump(status, file, indent=2)

    return status
