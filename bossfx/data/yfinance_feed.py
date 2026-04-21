"""
bossfx.data.yfinance_feed
=========================

YFinance wrapper for rapid prototyping.

⚠️ PRODUCTION WARNING ⚠️
yfinance is NOT a production FX data source. For EURUSD you're getting
unofficial aggregated data with small rounding mismatches in OHLC values
(open may be a hair above high, etc.). We sanitize those at the boundary
so the rest of the system can trust the data. For production, use MT5
historical or Dukascopy.
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
        self._sanitized_count = 0

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def timeframe(self) -> str:
        return self._timeframe

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        import yfinance as yf

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

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]

        rename_map = {}
        for candidate in ("date", "datetime", "index"):
            if candidate in df.columns:
                rename_map[candidate] = "timestamp"
                break
        df = df.rename(columns=rename_map)

        if "timestamp" not in df.columns:
            raise RuntimeError(
                f"Could not find date column. Got: {list(df.columns)}"
            )

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()

        self._df = df
        log.info(f"Downloaded {len(df)} bars for {self._symbol}")
        return df

    @staticmethod
    def _sanitize_ohlc(o: float, h: float, l: float, c: float) -> tuple[float, float, float, float]:
        """
        Fix minor OHLC inconsistencies that yfinance produces for FX pairs.
        Guarantees: low <= min(open, close, high) and high >= max(open, close, low).
        We never modify open or close — they're the critical values. We widen
        high/low to contain them if needed.
        """
        high = max(h, o, c, l)
        low = min(l, o, c, h)
        return o, high, low, c

    def stream(self) -> Iterator[BarEvent]:
        df = self._load()
        for row in df.itertuples(index=False):
            o, h, l, c = self._sanitize_ohlc(
                float(row.open), float(row.high), float(row.low), float(row.close)
            )
            orig = (float(row.open), float(row.high), float(row.low), float(row.close))
            if (o, h, l, c) != orig:
                self._sanitized_count += 1
            yield BarEvent(
                symbol=self._symbol,
                timestamp=row.timestamp.to_pydatetime(),
                open=o,
                high=h,
                low=l,
                close=c,
                volume=float(row.volume),
                timeframe=self._timeframe,
            )
        if self._sanitized_count:
            log.warning(
                f"Sanitized {self._sanitized_count} bars with OHLC inconsistencies "
                f"from yfinance. This is normal for FX; use a professional data "
                f"source for live trading."
            )
