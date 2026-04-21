"""
bossfx.strategies.sma_crossover
===============================

Classic SMA crossover — rebuilt properly.

Signal rules (Phase 1):
    * Fast SMA crosses ABOVE slow SMA  -> LONG
    * Fast SMA crosses BELOW slow SMA  -> SHORT
    * Otherwise -> HOLD (no signal emitted)

Why emit on the *cross*, not the *state*?
-----------------------------------------
The original bot emitted "buy" every bar where fast > slow. That means if
fast stayed above slow for 100 bars, we got 100 buy signals. Our portfolio
would either spam orders or need complex dedup logic. Emitting only on
crossover events is cleaner, more accurate to how traders actually think
about this strategy, and drops signal count by ~95%.

Phase 2 will layer trend/volatility/session filters on top of this core.
"""
from __future__ import annotations

from typing import Optional

from bossfx.core.events import BarEvent, SignalAction, SignalEvent
from bossfx.core.interfaces import Strategy
from bossfx.strategies.indicators import SMA
from bossfx.utils.logger import get_logger

log = get_logger(__name__)


class SMACrossoverStrategy(Strategy):
    strategy_id = "sma_crossover_v1"

    def __init__(self, fast_period: int = 20, slow_period: int = 50) -> None:
        if fast_period >= slow_period:
            raise ValueError("fast_period must be < slow_period")
        self._fast = SMA(fast_period)
        self._slow = SMA(slow_period)
        self._prev_diff: Optional[float] = None  # sign tracks last relationship

    def warmup_period(self) -> int:
        return self._slow.period

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        fast_val = self._fast.update(bar.close)
        slow_val = self._slow.update(bar.close)

        # Not warm yet
        if fast_val is None or slow_val is None:
            return None

        diff = fast_val - slow_val
        signal: Optional[SignalEvent] = None

        # Need a previous diff to detect a crossover
        if self._prev_diff is not None:
            # Bullish cross: prev <= 0, now > 0
            if self._prev_diff <= 0 and diff > 0:
                signal = SignalEvent(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    action=SignalAction.LONG,
                    strategy_id=self.strategy_id,
                    reference_price=bar.close,
                    metadata={"fast": fast_val, "slow": slow_val},
                )
            # Bearish cross: prev >= 0, now < 0
            elif self._prev_diff >= 0 and diff < 0:
                signal = SignalEvent(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    action=SignalAction.SHORT,
                    strategy_id=self.strategy_id,
                    reference_price=bar.close,
                    metadata={"fast": fast_val, "slow": slow_val},
                )

        self._prev_diff = diff
        if signal is not None:
            log.debug(f"{bar.timestamp} {signal.action.value} @ {bar.close}")
        return signal
