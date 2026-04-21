"""
Tests for SMACrossoverStrategy — Tier 3 (behavioral correctness).

Two things we must get right:
  1. Fires ONLY on the crossover bar — not every bar where fast > slow.
  2. Emits nothing during warmup.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from bossfx.core.events import BarEvent, SignalAction
from bossfx.strategies.sma_crossover import SMACrossoverStrategy


def _feed_closes(strat, closes, start=None):
    """Feed a list of close prices through the strategy, return emitted signals."""
    start = start or datetime(2024, 1, 1)
    signals = []
    for i, c in enumerate(closes):
        bar = BarEvent(
            symbol="EURUSD", timestamp=start + timedelta(hours=i),
            open=c, high=c + 0.001, low=c - 0.001, close=c, volume=100.0,
        )
        sig = strat.on_bar(bar)
        if sig is not None:
            signals.append((i, sig.action))
    return signals


class TestWarmup(unittest.TestCase):
    def test_no_signals_during_warmup(self):
        strat = SMACrossoverStrategy(fast_period=3, slow_period=5)
        signals = _feed_closes(strat, [1.0, 1.0, 1.0, 1.0])   # only 4 bars
        self.assertEqual(signals, [])

    def test_warmup_period_is_slow_period(self):
        strat = SMACrossoverStrategy(fast_period=3, slow_period=10)
        self.assertEqual(strat.warmup_period(), 10)


class TestCrossoverDetection(unittest.TestCase):
    def test_bullish_cross_emits_long_once(self):
        # Build a series where fast SMA clearly rises above slow SMA once
        #   First 5 bars flat at 1.0, then strong rise
        strat = SMACrossoverStrategy(fast_period=2, slow_period=4)
        closes = [1.0, 1.0, 1.0, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
        signals = _feed_closes(strat, closes)
        # Should fire LONG exactly once, and never fire again while trend holds
        longs = [s for s in signals if s[1] == SignalAction.LONG]
        self.assertEqual(len(longs), 1,
                         f"Expected exactly one LONG, got {signals}")

    def test_bearish_cross_emits_short_once(self):
        strat = SMACrossoverStrategy(fast_period=2, slow_period=4)
        closes = [1.0, 1.0, 1.0, 1.0, 0.99, 0.98, 0.97, 0.96, 0.95]
        signals = _feed_closes(strat, closes)
        shorts = [s for s in signals if s[1] == SignalAction.SHORT]
        self.assertEqual(len(shorts), 1)

    def test_flip_emits_both_signals(self):
        strat = SMACrossoverStrategy(fast_period=2, slow_period=4)
        # Rise, then fall
        closes = ([1.0] * 4 + [1.01, 1.02, 1.03, 1.04, 1.05]
                  + [1.00, 0.98, 0.95, 0.92, 0.89, 0.86, 0.83])
        signals = _feed_closes(strat, closes)
        actions = [s[1] for s in signals]
        self.assertIn(SignalAction.LONG, actions)
        self.assertIn(SignalAction.SHORT, actions)


class TestParameterValidation(unittest.TestCase):
    def test_rejects_fast_ge_slow(self):
        with self.assertRaises(ValueError):
            SMACrossoverStrategy(fast_period=20, slow_period=20)
        with self.assertRaises(ValueError):
            SMACrossoverStrategy(fast_period=50, slow_period=20)


if __name__ == "__main__":
    unittest.main()
