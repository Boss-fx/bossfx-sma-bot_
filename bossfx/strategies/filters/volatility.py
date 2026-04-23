"""
bossfx.strategies.filters.volatility
====================================

The ATR-based volatility regime filter.

Purpose
-------
Skip signals when the market is in a volatility regime that's hostile
to our fixed-percent stops/targets.
  * ATR too low (dead market)  -> moves too small to reach targets
  * ATR too high (shock regime) -> noise overwhelms our stop distance

Either extreme turns our 1:2 RR geometry into a losing coinflip, even
when the direction signal is correct. Better to sit out than to trade
into a regime that doesn't match our risk profile.

How it decides "in regime"
--------------------------
We compute ATR(period=14 default). Then we compute a slow SMA of that
ATR across the last N bars (default 100). The ratio

    current_ATR / avg_ATR

tells us where we are relative to this symbol's own recent normal:

  * ratio < 0.7  -> too quiet, veto
  * ratio > 1.5  -> too wild, veto
  * otherwise    -> pass (regime is reasonable)

Why relative-to-self rather than absolute?
------------------------------------------
Different pairs have wildly different ATR scales (GBPJPY can be 10x
EURUSD). Different timeframes too (1H ATR ~ 1/24 of daily). An absolute
threshold would need hand-tuning per pair + per timeframe. The relative
ratio is scale-free and self-calibrating.
"""

from __future__ import annotations

from typing import Optional

from bossfx.core.events import BarEvent, SignalAction, SignalEvent
from bossfx.strategies.indicators import ATR, SMA
from bossfx.utils.logger import get_logger

log = get_logger(__name__)


class ATRVolatilityFilter:
    """Vetoes signals when volatility is outside [min_ratio, max_ratio] range."""

    def __init__(
        self,
        atr_period: int = 14,
        lookback: int = 100,
        min_ratio: float = 0.7,
        max_ratio: float = 1.5,
    ) -> None:
        if atr_period <= 0 or lookback <= 0:
            raise ValueError("periods must be > 0")
        if min_ratio <= 0 or max_ratio <= min_ratio:
            raise ValueError("need 0 < min_ratio < max_ratio")
        self._atr = ATR(period=atr_period)
        self._atr_avg = SMA(period=lookback)
        self._last_atr: Optional[float] = None
        self._last_avg: Optional[float] = None
        self._atr_period = atr_period
        self._lookback = lookback
        self._min_ratio = min_ratio
        self._max_ratio = max_ratio
        self._vetoed_longs_low = 0
        self._vetoed_longs_high = 0
        self._vetoed_shorts_low = 0
        self._vetoed_shorts_high = 0
        self._passed_signals = 0

    @property
    def period(self) -> int:
        # Strategy uses this for warmup calc. We need enough bars to warm
        # both the ATR and the SMA-of-ATR.
        return self._atr_period + self._lookback

    def on_bar(self, bar: BarEvent) -> None:
        """Update ATR + its running average on EVERY bar."""
        atr_value = self._atr.update(bar.high, bar.low, bar.close)
        if atr_value is not None:
            self._last_atr = atr_value
            avg = self._atr_avg.update(atr_value)
            if avg is not None:
                self._last_avg = avg

    def allows(self, signal: SignalEvent) -> bool:
        """Return True if current vol is within tradeable range, else False."""
        # Exit signals always pass — closing is not subject to regime gating
        if signal.action == SignalAction.EXIT:
            self._passed_signals += 1
            return True

        # Not warm yet: conservative veto (same policy as TrendFilter)
        if self._last_atr is None or self._last_avg is None or self._last_avg == 0:
            log.debug(f"ATRVolatilityFilter VETO {signal.action.value}: not yet warm")
            self._count_veto(signal.action, "low")
            return False

        ratio = self._last_atr / self._last_avg
        too_low = ratio < self._min_ratio
        too_high = ratio > self._max_ratio

        if too_low:
            self._count_veto(signal.action, "low")
            log.debug(
                f"ATRVolatilityFilter VETO {signal.action.value}: "
                f"ratio={ratio:.3f} < min={self._min_ratio} (dead market)"
            )
            return False
        if too_high:
            self._count_veto(signal.action, "high")
            log.debug(
                f"ATRVolatilityFilter VETO {signal.action.value}: "
                f"ratio={ratio:.3f} > max={self._max_ratio} (shock regime)"
            )
            return False

        self._passed_signals += 1
        log.debug(f"ATRVolatilityFilter PASS {signal.action.value}: ratio={ratio:.3f}")
        return True

    def _count_veto(self, action: SignalAction, kind: str) -> None:
        if action == SignalAction.LONG:
            if kind == "low":
                self._vetoed_longs_low += 1
            else:
                self._vetoed_longs_high += 1
        elif action == SignalAction.SHORT:
            if kind == "low":
                self._vetoed_shorts_low += 1
            else:
                self._vetoed_shorts_high += 1

    def stats(self) -> dict:
        total_low = self._vetoed_longs_low + self._vetoed_shorts_low
        total_high = self._vetoed_longs_high + self._vetoed_shorts_high
        total_vetoed = total_low + total_high
        total_seen = total_vetoed + self._passed_signals
        return {
            "atr_period": self._atr_period,
            "lookback": self._lookback,
            "min_ratio": self._min_ratio,
            "max_ratio": self._max_ratio,
            "passed": self._passed_signals,
            "vetoed_low_vol": total_low,
            "vetoed_high_vol": total_high,
            "total_vetoed": total_vetoed,
            "total_seen": total_seen,
            "veto_rate": (total_vetoed / total_seen) if total_seen else 0.0,
        }
