"""
bossfx.data.yfinance_feed
=========================

YFinance wrapper for rapid prototyping.

Uses yf.Ticker().history() instead of yf.download() because Yahoo's
bulk-download endpoint has become unreliable for forex symbols (HTTP 500
"sad panda" responses). The Ticker endpoint hits a different API path
that remains functional.

Still not a production data source for live trading. See warning below.

PRODUCTION WARNING
------------------
yfinance is NOT a production FX data source. For EURUSD you're getting
unofficial aggregated data with small rounding mismatches in OHLC values.
We sanitize those at the boundary so the rest of the system can trust
the data. For production, use MT5 historical or Dukascopy.
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

        log.info(
            f"Downloading {self._symbol} {self._timeframe} from yfinance "
            f"(Ticker path) [{self._start} -> {self._end}]..."
        )
        ticker = yf.Ticker(self._symbol)
        df = ticker.history(
            start=self._start,
            end=self._end,
            interval=self._timeframe,
            auto_adjust=False,
        )

        if df is None or df.empty:
            raise RuntimeError(
                f"yfinance returned no data for {self._symbol} "
                f"[{self._start} -> {self._end}] at interval {self._timeframe}. "
                f"Note: 1h data is limited to the trailing 730 days from today."
            )

        # Ticker().history() returns a DatetimeIndex named 'Datetime'.
        # reset_index() promotes it into a column; we then normalise.
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]

        # The date column can be 'datetime', 'date', or 'index' depending on
        # interval + yfinance version. Normalise to 'timestamp'.
        rename_map = {}
        for candidate in ("datetime", "date", "index"):
            if candidate in df.columns:
                rename_map[candidate] = "timestamp"
                break
        df = df.rename(columns=rename_map)

        if "timestamp" not in df.columns:
            raise RuntimeError(f"Could not find date column. Got: {list(df.columns)}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()

        self._df = df
        log.info(f"Downloaded {len(df)} bars for {self._symbol}")
        return df

    @staticmethod
    def _sanitize_ohlc(
        open_price: float, high: float, low: float, close: float
    ) -> tuple[float, float, float, float]:
        """
        Fix minor OHLC inconsistencies that yfinance produces for FX pairs.
        We never modify open or close - they're the critical values. We
        widen high/low to contain them if needed.
        """
        new_high = max(high, open_price, close, low)
        new_low = min(low, open_price, close, high)
        return open_price, new_high, new_low, close

    def stream(self) -> Iterator[BarEvent]:
        df = self._load()
        for row in df.itertuples(index=False):
            open_price, high, low, close = self._sanitize_ohlc(
                float(row.open), float(row.high), float(row.low), float(row.close)
            )
            orig = (float(row.open), float(row.high), float(row.low), float(row.close))
            if (open_price, high, low, close) != orig:
                self._sanitized_count += 1
            yield BarEvent(
                symbol=self._symbol,
                timestamp=row.timestamp.to_pydatetime(),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=float(row.volume),
                timeframe=self._timeframe,
            )
        if self._sanitized_count:
            log.warning(
                f"Sanitized {self._sanitized_count} bars with OHLC inconsistencies "
                f"from yfinance. This is normal for FX; use a professional data "
                f"source for live trading."
            )
