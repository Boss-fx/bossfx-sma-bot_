"""
bossfx.strategies.indicators
============================

*Online* (streaming, stateful) indicators.

Why online?
-----------
A vectorized pandas ``df['close'].rolling(20).mean()`` is fast - and
dangerous. Online indicators receive one value at a time and can only
return results based on values already seen. Look-ahead bias becomes
structurally impossible.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional


class SMA:
    """Simple Moving Average - online, O(1) per update."""

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be > 0")
        self.period = period
        self._buf: Deque[float] = deque(maxlen=period)
        self._sum = 0.0

    def update(self, value: float) -> Optional[float]:
        """Push a new value. Returns the SMA if warm, else None."""
        if len(self._buf) == self.period:
            self._sum -= self._buf[0]
        self._buf.append(value)
        self._sum += value
        if len(self._buf) < self.period:
            return None
        return self._sum / self.period

    @property
    def is_warm(self) -> bool:
        return len(self._buf) == self.period

    @property
    def value(self) -> Optional[float]:
        return self._sum / self.period if self.is_warm else None


class EMA:
    """
    Exponential Moving Average - online, O(1) per update.

    Why EMA instead of SMA for trend detection?
    -------------------------------------------
    The SMA gives equal weight to every value in its window. A big price
    move from 200 bars ago counts just as much as yesterday's close. For
    trend detection we want the OPPOSITE: recent data should dominate.

    The EMA does this by weighting each value as a decaying exponential:
        today = (1-alpha) * yesterday + alpha * today's value
    (For period=200, alpha ~ 0.01 - recent data still dominates over months.)

    The smoothing factor ``alpha = 2 / (period + 1)`` is the standard
    TradingView / MT4 / institutional convention. Your EMA(200) in BossFx
    will match what you see on any charting platform.
    """

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be > 0")
        self.period = period
        self._alpha = 2.0 / (period + 1)
        self._value: Optional[float] = None
        # We seed with SMA of the first `period` values for a stable start.
        # This matches the TradingView / MT4 convention.
        self._seed_buf: Deque[float] = deque(maxlen=period)

    def update(self, value: float) -> Optional[float]:
        """Push a new value. Returns the EMA if warm, else None."""
        if self._value is None:
            self._seed_buf.append(value)
            if len(self._seed_buf) < self.period:
                return None
            # Just reached warm - seed EMA with SMA of the buffer
            self._value = sum(self._seed_buf) / self.period
            return self._value

        # Already warm - apply the EMA recurrence
        self._value = self._alpha * value + (1 - self._alpha) * self._value
        return self._value

    @property
    def is_warm(self) -> bool:
        return self._value is not None

    @property
    def value(self) -> Optional[float]:
        return self._value


class ATR:
    """Average True Range - online. Uses Wilder's smoothing."""

    def __init__(self, period: int = 14) -> None:
        if period <= 0:
            raise ValueError("period must be > 0")
        self.period = period
        self._trs: Deque[float] = deque(maxlen=period)
        self._prev_close: Optional[float] = None
        self._atr: Optional[float] = None

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
        self._prev_close = close

        if self._atr is None:
            self._trs.append(tr)
            if len(self._trs) == self.period:
                self._atr = sum(self._trs) / self.period
            return self._atr
        # Wilder smoothing
        self._atr = (self._atr * (self.period - 1) + tr) / self.period
        return self._atr

    @property
    def value(self) -> Optional[float]:
        return self._atr
