import unittest

import pandas as pd


class _FakeResult:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_df(self) -> pd.DataFrame:
        return self._df


class _FakePriceRouter:
    def __init__(self, rows: list[dict], fail_times: int = 0):
        self.rows = rows
        self.fail_times = fail_times
        self.calls = []

    def historical(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("temporary failure")
        df = pd.DataFrame(self.rows)
        if "date" in df.columns:
            df = df.set_index("date")
            df.index.name = "date"
        return _FakeResult(df)


class _FakeNewsRouter:
    def __init__(self):
        self.calls = []

    def company(self, **kwargs):
        self.calls.append(kwargs)
        df = pd.DataFrame(
            [
                {
                    "date": "2026-03-01T00:00:00Z",
                    "title": "Headline",
                    "url": "https://example.com",
                    "source": "Source",
                    "symbol": "^HSI",
                    "summary": "Summary",
                    "text": "Text",
                    "id": "id-1",
                }
            ]
        ).set_index("date")
        df.index.name = "date"
        return _FakeResult(df)


class _FakeOBB:
    def __init__(self, index_router, equity_router, currency_router, news_router):
        self.index = type("Index", (), {"price": index_router})()
        self.equity = type("Equity", (), {"price": equity_router})()
        self.currency = type("Currency", (), {"price": currency_router})()
        self.news = news_router


class TestOpenBBClient(unittest.TestCase):
    def test_normalizes_single_symbol_when_symbol_column_missing(self):
        from src.data.openbb_client import OpenBBClient

        router = _FakePriceRouter(
            rows=[
                {"date": "2026-02-27", "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 10},
                {"date": "2026-02-28", "open": 1.1, "high": 1.3, "low": 1.0, "close": 1.2, "volume": 11},
            ]
        )
        fake_obb = _FakeOBB(index_router=router, equity_router=router, currency_router=router, news_router=_FakeNewsRouter())
        client = OpenBBClient(provider="yfinance", obb_module=fake_obb, retries=1, retry_delay=0.0)

        df = client.fetch_index_history(symbols=["^HSI"], start_date="2026-02-27", end_date="2026-02-28")

        self.assertIn("symbol", df.columns)
        self.assertEqual(df["symbol"].nunique(), 1)
        self.assertEqual(df["symbol"].iloc[0], "^HSI")
        self.assertIn("date", df.columns)

    def test_forwards_provider_to_openbb_requests(self):
        from src.data.openbb_client import OpenBBClient

        router = _FakePriceRouter(rows=[{"date": "2026-02-27", "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 10}])
        fake_obb = _FakeOBB(index_router=router, equity_router=router, currency_router=router, news_router=_FakeNewsRouter())
        client = OpenBBClient(provider="yfinance", obb_module=fake_obb, retries=1, retry_delay=0.0)

        client.fetch_equity_history(symbols=["0700.HK"], start_date="2026-02-27", end_date="2026-02-28")

        self.assertEqual(router.calls[-1]["provider"], "yfinance")
        self.assertEqual(router.calls[-1]["symbol"], ["0700.HK"])

    def test_retries_then_succeeds(self):
        from src.data.openbb_client import OpenBBClient

        retry_router = _FakePriceRouter(
            rows=[{"date": "2026-02-27", "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 10}],
            fail_times=2,
        )
        fake_obb = _FakeOBB(
            index_router=retry_router,
            equity_router=retry_router,
            currency_router=retry_router,
            news_router=_FakeNewsRouter(),
        )
        client = OpenBBClient(provider="yfinance", obb_module=fake_obb, retries=3, retry_delay=0.0)

        df = client.fetch_currency_history(symbols=["USDHKD"], start_date="2026-02-27", end_date="2026-02-28")

        self.assertEqual(len(df), 1)
        self.assertEqual(len(retry_router.calls), 3)


if __name__ == "__main__":
    unittest.main()
