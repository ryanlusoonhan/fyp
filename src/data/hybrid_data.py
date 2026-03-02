from __future__ import annotations

import os
import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import OPENBB_NEWS_FILE, OPENBB_TRAINING_FILE, PROCESSED_DATA_PATH

SECONDARY_MARKET_COLS = [
    "VIX_Close",
    "TNX_Close",
    "GSPC_Close",
    "USDHKD_Close",
    "USDCNY_Close",
    "HK_Breadth_Positive_1D",
    "HK_Breadth_Above_MA20",
    "HK_Breadth_Dispersion_5D",
]

POSITIVE_TOKENS = {
    "beat",
    "beats",
    "bull",
    "bullish",
    "gain",
    "gains",
    "growth",
    "improve",
    "improves",
    "improved",
    "optimism",
    "outperform",
    "outperforms",
    "profit",
    "profits",
    "rally",
    "rallies",
    "rebound",
    "recovery",
    "strong",
    "surge",
    "surges",
    "upside",
}

NEGATIVE_TOKENS = {
    "bear",
    "bearish",
    "concern",
    "concerns",
    "decline",
    "declines",
    "drop",
    "drops",
    "fall",
    "falls",
    "loss",
    "losses",
    "miss",
    "misses",
    "risk",
    "risks",
    "slump",
    "slumps",
    "slowdown",
    "weak",
    "warning",
    "warnings",
}


def _normalize_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    parsed = parsed.dt.tz_convert(None)
    return parsed.dt.floor("D")


def _prepare_primary_df(primary_df: pd.DataFrame) -> pd.DataFrame:
    out = primary_df.copy()
    if "Date" not in out.columns:
        raise ValueError("Primary dataset must contain `Date` column.")
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return out


def _prepare_secondary_market_df(secondary_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if secondary_df is None or len(secondary_df) == 0:
        return None

    out = secondary_df.copy()
    if "Date" not in out.columns:
        if "date" in out.columns:
            out = out.rename(columns={"date": "Date"})
        else:
            return None

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    available = ["Date", *[col for col in SECONDARY_MARKET_COLS if col in out.columns]]
    out = out[available]
    if len(out.columns) <= 1:
        return None
    return out


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _lexicon_sentiment_score(text: str) -> float:
    tokens = _tokenize(text)
    if len(tokens) == 0:
        return 0.0
    pos = sum(1 for token in tokens if token in POSITIVE_TOKENS)
    neg = sum(1 for token in tokens if token in NEGATIVE_TOKENS)
    if pos == 0 and neg == 0:
        return 0.0
    return float(pos - neg) / float(pos + neg)


def compute_openbb_news_sentiment(news_df: pd.DataFrame | None) -> pd.DataFrame:
    if news_df is None or len(news_df) == 0:
        return pd.DataFrame(columns=["Date", "OpenBB_News_Sentiment", "OpenBB_News_Article_Count"])

    out = news_df.copy()
    date_col = "date" if "date" in out.columns else ("Date" if "Date" in out.columns else None)
    if date_col is None:
        return pd.DataFrame(columns=["Date", "OpenBB_News_Sentiment", "OpenBB_News_Article_Count"])

    out["Date"] = _normalize_date_series(out[date_col])
    out = out.dropna(subset=["Date"]).copy()
    if len(out) == 0:
        return pd.DataFrame(columns=["Date", "OpenBB_News_Sentiment", "OpenBB_News_Article_Count"])

    for col in ["title", "summary", "text"]:
        if col not in out.columns:
            out[col] = ""
    out["news_text"] = (
        out["title"].fillna("").astype(str)
        + " "
        + out["summary"].fillna("").astype(str)
        + " "
        + out["text"].fillna("").astype(str)
    ).str.strip()
    out["sentiment"] = out["news_text"].map(_lexicon_sentiment_score)

    grouped = (
        out.groupby("Date", as_index=False)
        .agg(
            OpenBB_News_Sentiment=("sentiment", "mean"),
            OpenBB_News_Article_Count=("sentiment", "size"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )
    grouped["OpenBB_News_Sentiment"] = grouped["OpenBB_News_Sentiment"].astype(float)
    grouped["OpenBB_News_Article_Count"] = grouped["OpenBB_News_Article_Count"].astype(float)
    return grouped


def _ensure_secondary_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = 0.0
    return out


def build_hybrid_training_dataset(
    primary_df: pd.DataFrame,
    secondary_market_df: pd.DataFrame | None = None,
    news_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    primary = _prepare_primary_df(primary_df)
    secondary = _prepare_secondary_market_df(secondary_market_df)

    merged = primary.copy()
    if secondary is not None:
        merged = merged.merge(secondary, on="Date", how="left")
    merged = _ensure_secondary_columns(merged, SECONDARY_MARKET_COLS)
    merged[SECONDARY_MARKET_COLS] = merged[SECONDARY_MARKET_COLS].ffill().fillna(0.0)

    news_daily = compute_openbb_news_sentiment(news_df)
    merged = merged.merge(news_daily, on="Date", how="left")
    merged["OpenBB_News_Sentiment"] = pd.to_numeric(
        merged.get("OpenBB_News_Sentiment", 0.0), errors="coerce"
    ).fillna(0.0)
    merged["OpenBB_News_Article_Count"] = pd.to_numeric(
        merged.get("OpenBB_News_Article_Count", 0.0), errors="coerce"
    ).fillna(0.0)
    merged["OpenBB_News_Availability"] = (merged["OpenBB_News_Article_Count"] > 0).astype(float)

    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


def load_hybrid_training_dataset(
    primary_path: str | None = None,
    secondary_path: str = OPENBB_TRAINING_FILE,
    news_path: str = OPENBB_NEWS_FILE,
) -> pd.DataFrame:
    primary = primary_path or f"{PROCESSED_DATA_PATH}training_data.csv"
    if not os.path.exists(primary):
        raise FileNotFoundError(f"Primary training dataset not found: {primary}")

    primary_df = pd.read_csv(primary)

    secondary_df = None
    if secondary_path and os.path.exists(secondary_path):
        same_file = os.path.normpath(primary) == os.path.normpath(secondary_path)
        if not same_file:
            try:
                same_file = os.path.samefile(primary, secondary_path)
            except OSError:
                same_file = False
        if not same_file:
            secondary_df = pd.read_csv(secondary_path)

    news_df = pd.read_csv(news_path) if news_path and os.path.exists(news_path) else None
    return build_hybrid_training_dataset(primary_df=primary_df, secondary_market_df=secondary_df, news_df=news_df)
