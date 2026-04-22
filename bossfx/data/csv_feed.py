"""
bossfx.data.csv_feed
====================

A CSV-backed data feed. Essential because:
  1. Reproducible tests need deterministic data.
  2. Downloaded data can be cached as CSV/Parquet to avoid hammering APIs.
  3. Many brokers export history as CSV — this is how you onboard real data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from bossfx.core.events import BarEvent
from bossfx.core.interfaces import DataFeed
from bossfx.utils.logger import get_logger

log = get_logger(__name__)


class CSVDataFeed(DataFeed):
    """
    Expects a CSV with columns: timestamp, open, high, low, close, volume.
    Timestamp column is parsed as UTC.
    """

    def __init__(
        self,
        path: str | Path,
        symbol: str,
        timeframe: str = "1H",
    ) -> None:
        self._path = Path(path)
        self._symbol = symbol
        self._timeframe = timeframe

        if not self._path.exists():
            raise FileNotFoundError(f"CSV not found: {self._path}")

        df = pd.read_csv(self._path)
        df.columns = [c.lower() for c in df.columns]
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        self._df = df
        log.info(f"Loaded {len(df)} bars for {symbol} from {self._path.name}")

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def timeframe(self) -> str:
        return self._timeframe

    def stream(self) -> Iterator[BarEvent]:
        for row in self._df.itertuples(index=False):
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

    def to_dataframe(self) -> pd.DataFrame:
        """Escape hatch for tools that need the whole frame (e.g., warmup)."""
        return self._df.copy()
