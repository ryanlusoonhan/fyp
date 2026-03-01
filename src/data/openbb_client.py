from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


class OpenBBClientError(RuntimeError):
    pass


class OpenBBRequestError(OpenBBClientError):
    pass


class OpenBBDataError(OpenBBClientError):
    pass


@dataclass
class OpenBBRequestDiagnostics:
    endpoint: str
    provider: str
    symbols: list[str]
    attempts: int
    rows: int


class OpenBBClient:
    def __init__(
        self,
        provider: str = "yfinance",
        retries: int = 3,
        retry_delay: float = 1.0,
        obb_module: Any | None = None,
    ) -> None:
        self.provider = provider
        self.retries = max(1, int(retries))
        self.retry_delay = max(0.0, float(retry_delay))
        self.last_diagnostics: OpenBBRequestDiagnostics | None = None
        self._obb = obb_module if obb_module is not None else self._load_obb()

    @staticmethod
    def _load_obb():
        try:
            from openbb import obb
        except Exception as exc:  # pragma: no cover - import path depends on local env
            raise OpenBBClientError(
                "OpenBB is not installed. Install `openbb` and `openbb-yfinance`."
            ) from exc
        return obb

    def _execute_with_retry(self, fn: Callable[[], Any], endpoint: str) -> tuple[Any, int]:
        attempts = 0
        last_error: Exception | None = None
        for attempts in range(1, self.retries + 1):
            try:
                return fn(), attempts
            except Exception as exc:  # noqa: PERF203
                last_error = exc
                if attempts >= self.retries:
                    break
                if self.retry_delay > 0:
                    time.sleep(self.retry_delay * attempts)
        raise OpenBBRequestError(f"OpenBB request failed for {endpoint}: {last_error}") from last_error

    @staticmethod
    def _normalize_price_df(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
        out = df.copy()
        if "date" not in out.columns:
            if out.index.name == "date" or isinstance(out.index, (pd.DatetimeIndex, pd.Index)):
                out = out.reset_index()
        if "date" not in out.columns:
            raise OpenBBDataError("OpenBB response is missing `date` column.")

        out.columns = [str(col).lower() for col in out.columns]
        out = out.rename(columns={"adj close": "close"})

        needed = {"date", "open", "high", "low", "close", "volume"}
        missing = [col for col in needed if col not in out.columns]
        if missing:
            raise OpenBBDataError(f"OpenBB price response is missing columns: {missing}")

        if "symbol" not in out.columns:
            if len(symbols) == 1:
                out["symbol"] = symbols[0]
            else:
                raise OpenBBDataError("OpenBB response is missing `symbol` for multi-symbol request.")

        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.tz_convert(None)
        out = out.dropna(subset=["date"]).sort_values(["date", "symbol"]).reset_index(drop=True)
        return out[list(needed) + ["symbol"]]

    @staticmethod
    def _normalize_news_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "date" not in out.columns:
            if out.index.name == "date" or isinstance(out.index, (pd.DatetimeIndex, pd.Index)):
                out = out.reset_index()
        if "date" not in out.columns:
            raise OpenBBDataError("OpenBB news response is missing `date` column.")

        out.columns = [str(col).lower() for col in out.columns]
        for col in ["title", "url", "source", "symbol", "summary", "text", "id"]:
            if col not in out.columns:
                out[col] = None
        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return out[["date", "title", "url", "source", "symbol", "summary", "text", "id"]]

    def _fetch_price(
        self,
        endpoint: str,
        fetcher: Callable[..., Any],
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        args = {
            "symbol": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "provider": self.provider,
        }
        result, attempts = self._execute_with_retry(lambda: fetcher(**args), endpoint)
        df = self._normalize_price_df(result.to_df(), symbols=symbols)
        self.last_diagnostics = OpenBBRequestDiagnostics(
            endpoint=endpoint,
            provider=self.provider,
            symbols=symbols,
            attempts=attempts,
            rows=len(df),
        )
        return df

    def fetch_index_history(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        return self._fetch_price(
            endpoint="index.price.historical",
            fetcher=self._obb.index.price.historical,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

    def fetch_equity_history(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        return self._fetch_price(
            endpoint="equity.price.historical",
            fetcher=self._obb.equity.price.historical,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

    def fetch_currency_history(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        return self._fetch_price(
            endpoint="currency.price.historical",
            fetcher=self._obb.currency.price.historical,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

    def fetch_recent_news(self, symbols: list[str], limit: int = 100) -> pd.DataFrame:
        args = {
            "symbol": symbols,
            "limit": int(limit),
            "provider": self.provider,
        }
        result, attempts = self._execute_with_retry(
            lambda: self._obb.news.company(**args),
            "news.company",
        )
        df = self._normalize_news_df(result.to_df())
        self.last_diagnostics = OpenBBRequestDiagnostics(
            endpoint="news.company",
            provider=self.provider,
            symbols=symbols,
            attempts=attempts,
            rows=len(df),
        )
        return df
