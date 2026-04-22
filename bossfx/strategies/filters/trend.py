"""
bossfx.strategies.filters.trend
===============================

The higher-timeframe trend filter.

Purpose
-------
Veto signals that go against the prevailing trend. A professional trader
never takes a countertrend trade on a first-order momentum signal like
SMA crossover - the edge simply isn't there on average.

How it decides "trend"
----------------------
We compare the current close price against an EMA (default period 200).
  * Close ABOVE EMA   -> trend is UP   -> allow LONG, veto SHORT
  * Close BELOW EMA   -> trend is DOWN -> allow SHORT, veto LONG
  * EMA not yet warm  -> trend unknown -> veto everything (conservative)
"""

from __future__ import annotations

from typing import Optional

from bossfx.core.events import BarEvent, SignalAction, SignalEvent
from bossfx.strategies.indicators import EMA
from bossfx.utils.logger import get_logger

log = get_logger(__name__)


class TrendFilter:
    """Vetoes signals that go against the EMA-defined trend."""

    def __init__(self, period: int = 200) -> None:
        if period <= 0:
            raise ValueError("period must be > 0")
        self._ema = EMA(period)
        self._last_close: Optional[float] = None
        self._period = period
        self._vetoed_longs = 0
        self._vetoed_shorts = 0
        self._passed_signals = 0

    @property
    def period(self) -> int:
        return self._period

    def on_bar(self, bar: BarEvent) -> None:
        """Update internal state. Must be called on EVERY bar, not just signals."""
        self._ema.update(bar.close)
        self._last_close = bar.close

    def allows(self, signal: SignalEvent) -> bool:
        """Return True if the signal should be passed to risk manager, else False."""
        if signal.action == SignalAction.EXIT:
            self._passed_signals += 1
            return True

        if not self._ema.is_warm or self._last_close is None:
            log.debug(f"TrendFilter VETO {signal.action.value}: EMA({self._period}) not yet warm")
            self._count_veto(signal.action)
            return False

        ema_value = self._ema.value
        trend_is_up = self._last_close > ema_value

        allowed = (signal.action == SignalAction.LONG and trend_is_up) or (
            signal.action == SignalAction.SHORT and not trend_is_up
        )

        if allowed:
            self._passed_signals += 1
            log.debug(
                f"TrendFilter PASS {signal.action.value}: "
                f"close={self._last_close:.5f} vs EMA={ema_value:.5f}"
            )
        else:
            self._count_veto(signal.action)
            log.debug(
                f"TrendFilter VETO {signal.action.value}: "
                f"close={self._last_close:.5f} vs EMA={ema_value:.5f} "
                f"(trend_up={trend_is_up})"
            )

        return allowed

    def _count_veto(self, action: SignalAction) -> None:
        if action == SignalAction.LONG:
            self._vetoed_longs += 1
        elif action == SignalAction.SHORT:
            self._vetoed_shorts += 1

    def stats(self) -> dict:
        """Stats for analytics - how many signals did the filter eat?"""
        total_vetoed = self._vetoed_longs + self._vetoed_shorts
        total_seen = total_vetoed + self._passed_signals
        return {
            "period": self._period,
            "passed": self._passed_signals,
            "vetoed_longs": self._vetoed_longs,
            "vetoed_shorts": self._vetoed_shorts,
            "total_seen": total_seen,
            "veto_rate": (total_vetoed / total_seen) if total_seen else 0.0,
        }
