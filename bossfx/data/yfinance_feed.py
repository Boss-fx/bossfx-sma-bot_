"""
bossfx.data.yfinance_feed
=========================

YFinance wrapper for rapid prototyping.

⚠️ PRODUCTION WARNING ⚠️
yfinance is NOT a production FX data source. For EURUSD you're getting
unofficial aggregated data with questionable timestamps. Use it to
iterate, then replace with:
  * MT5 historical API (free, broker-accurate)
  * Dukascopy tick data (free, institutional-grade)
  * Polygon / Databento (paid, rock-solid)

The beauty of our design: swapping is a 10-line change.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterator, Optional

import pandas as pd

from bossfx.core.events import BarEvent
from bossfx.core.interfaces import DataFeed
from bossfx.utils.logger import get_logger

log = get_logger(__name__)


class YFinanceDataFeed(DataFeed):
    def __init__(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1h",
    ) -> None:
        self._symbol = symbol
        self._timeframe = interval
        self._start = start
        self._end = end
        self._df: Optional[pd.DataFrame] = None

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def timeframe(self) -> str:
        return self._timeframe

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        import yfinance as yf  # lazy import so tests don't need network

        log.info(f"Downloading {self._symbol} {self._timeframe} from yfinance...")
        df = yf.download(
            tickers=self._symbol,
            start=self._start,
            end=self._end,
            interval=self._timeframe,
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            raise RuntimeError(f"yfinance returned no data for {self._symbol}")

        # Flatten columns if multi-indexed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns=str.lower).reset_index()
        df = df.rename(columns={"datetime": "timestamp", "date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()

        self._df = df
        log.info(f"Downloaded {len(df)} bars for {self._symbol}")
        return df

    def stream(self) -> Iterator[BarEvent]:
        df = self._load()
        for row in df.itertuples(index=False):
            yield BarEvent(
                symbol=self._symbol,
                timestamp=row.timestamp.to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                timeframe=self._timeframe,
            )
