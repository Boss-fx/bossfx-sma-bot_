"""
bossfx.strategies.sma_crossover
===============================

Classic SMA crossover - now with optional filter composition.
"""

from __future__ import annotations

from typing import List, Optional, Protocol

from bossfx.core.events import BarEvent, SignalAction, SignalEvent
from bossfx.core.interfaces import Strategy
from bossfx.strategies.indicators import SMA
from bossfx.utils.logger import get_logger

log = get_logger(__name__)


class SignalFilter(Protocol):
    """Structural contract any filter must honor. No inheritance required."""

    def on_bar(self, bar: BarEvent) -> None: ...
    def allows(self, signal: SignalEvent) -> bool: ...


class SMACrossoverStrategy(Strategy):
    strategy_id = "sma_crossover_v2"

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        filters: Optional[List[SignalFilter]] = None,
    ) -> None:
        if fast_period >= slow_period:
            raise ValueError("fast_period must be < slow_period")
        self._fast = SMA(fast_period)
        self._slow = SMA(slow_period)
        self._prev_diff: Optional[float] = None
        self._filters: List[SignalFilter] = filters or []

    def warmup_period(self) -> int:
        base = self._slow.period
        for f in self._filters:
            f_period = getattr(f, "period", 0)
            if isinstance(f_period, int) and f_period > base:
                base = f_period
        return base

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        # Update all filters every bar - they need continuous state.
        for f in self._filters:
            f.on_bar(bar)

        fast_val = self._fast.update(bar.close)
        slow_val = self._slow.update(bar.close)

        if fast_val is None or slow_val is None:
            return None

        diff = fast_val - slow_val
        raw_signal: Optional[SignalEvent] = None

        if self._prev_diff is not None:
            if self._prev_diff <= 0 and diff > 0:
                raw_signal = SignalEvent(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    action=SignalAction.LONG,
                    strategy_id=self.strategy_id,
                    reference_price=bar.close,
                    metadata={"fast": fast_val, "slow": slow_val},
                )
            elif self._prev_diff >= 0 and diff < 0:
                raw_signal = SignalEvent(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    action=SignalAction.SHORT,
                    strategy_id=self.strategy_id,
                    reference_price=bar.close,
                    metadata={"fast": fast_val, "slow": slow_val},
                )

        self._prev_diff = diff

        if raw_signal is None:
            return None

        # Run through filter chain. Any veto drops the signal.
        for f in self._filters:
            if not f.allows(raw_signal):
                return None

        log.debug(f"{bar.timestamp} {raw_signal.action.value} @ {bar.close}")
        return raw_signal
