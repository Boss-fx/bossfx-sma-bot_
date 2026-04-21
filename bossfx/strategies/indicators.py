"""
bossfx.strategies.indicators
============================

*Online* (a.k.a. streaming, stateful) indicators.

Why online?
-----------
A vectorized pandas ``df['close'].rolling(20).mean()`` is fast — and
dangerous. It's a single computation over the whole series, which means
when you later shift/compare, it's easy to accidentally compare today's
SMA with today's close (and today's SMA uses today's close to compute!).
That's a subtle look-ahead bias.

Online indicators receive one value at a time and can only return results
based on values already seen. If the strategy can't compute tomorrow's
SMA, it can't cheat with it. The bias becomes structurally impossible.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional


class SMA:
    """Simple Moving Average — online, O(1) per update."""

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be > 0")
        self.period = period
        self._buf: Deque[float] = deque(maxlen=period)
        self._sum = 0.0

    def update(self, value: float) -> Optional[float]:
        """Push a new value. Returns the SMA if warm, else None."""
        if len(self._buf) == self.period:
            self._sum -= self._buf[0]   # evicted when we append below
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


class ATR:
    """Average True Range — online. Uses Wilder's smoothing."""

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
