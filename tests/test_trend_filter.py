"""
Tests for the EMA indicator and TrendFilter - Phase 2.1.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from bossfx.core.events import BarEvent, SignalAction, SignalEvent
from bossfx.strategies.filters.trend import TrendFilter
from bossfx.strategies.indicators import EMA


def _bar(close, ts=None):
    ts = ts or datetime(2024, 1, 1)
    return BarEvent(
        symbol="EURUSD",
        timestamp=ts,
        open=close,
        high=close + 0.001,
        low=close - 0.001,
        close=close,
        volume=100.0,
    )


def _signal(action):
    return SignalEvent(
        symbol="EURUSD",
        timestamp=datetime(2024, 1, 1),
        action=action,
        strategy_id="test",
    )


class TestEMABasic(unittest.TestCase):
    def test_returns_none_before_warm(self):
        e = EMA(period=3)
        self.assertIsNone(e.update(1.0))
        self.assertIsNone(e.update(2.0))
        self.assertIsNotNone(e.update(3.0))

    def test_seed_equals_sma_of_first_period(self):
        e = EMA(period=5)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = None
        for v in values:
            result = e.update(v)
        expected = sum(values) / 5
        self.assertAlmostEqual(result, expected, places=10)

    def test_rejects_zero_or_negative_period(self):
        with self.assertRaises(ValueError):
            EMA(period=0)
        with self.assertRaises(ValueError):
            EMA(period=-5)

    def test_alpha_matches_institutional_convention(self):
        e = EMA(period=10)
        self.assertAlmostEqual(e._alpha, 2.0 / 11.0, places=10)

    def test_ema_recurrence_after_seed(self):
        e = EMA(period=3)
        for v in [1.0, 2.0, 3.0]:
            e.update(v)
        seed = e.value
        alpha = 2.0 / 4.0

        next_val = e.update(10.0)
        expected = alpha * 10.0 + (1 - alpha) * seed
        self.assertAlmostEqual(next_val, expected, places=10)


class TestEMANoLookAhead(unittest.TestCase):
    def test_future_data_cannot_alter_past_values(self):
        prefix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        e1 = EMA(period=3)
        for v in prefix:
            e1.update(v)
        baseline = e1.value

        e2 = EMA(period=3)
        for v in prefix:
            e2.update(v)
        for v in [1000.0, -500.0]:
            e2.update(v)

        e3 = EMA(period=3)
        for v in prefix:
            e3.update(v)
        recomputed = e3.value

        self.assertAlmostEqual(baseline, recomputed, places=10)


class TestTrendFilterWarmup(unittest.TestCase):
    def test_vetos_all_signals_before_warm(self):
        f = TrendFilter(period=5)
        for i, price in enumerate([1.10, 1.11, 1.12]):
            f.on_bar(_bar(price, datetime(2024, 1, 1) + timedelta(hours=i)))
        self.assertFalse(f.allows(_signal(SignalAction.LONG)))
        self.assertFalse(f.allows(_signal(SignalAction.SHORT)))

    def test_exit_signals_always_allowed_even_before_warm(self):
        f = TrendFilter(period=5)
        self.assertTrue(f.allows(_signal(SignalAction.EXIT)))


class TestTrendFilterDecision(unittest.TestCase):
    def _warmed_filter(self, prices):
        f = TrendFilter(period=len(prices))
        for i, p in enumerate(prices):
            f.on_bar(_bar(p, datetime(2024, 1, 1) + timedelta(hours=i)))
        return f

    def test_long_allowed_when_close_above_ema(self):
        f = self._warmed_filter([1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06])
        f.on_bar(_bar(1.20, datetime(2024, 1, 2)))
        self.assertTrue(f.allows(_signal(SignalAction.LONG)))
        self.assertFalse(f.allows(_signal(SignalAction.SHORT)))

    def test_short_allowed_when_close_below_ema(self):
        f = self._warmed_filter([1.10, 1.09, 1.08, 1.07, 1.06, 1.05, 1.04])
        f.on_bar(_bar(0.90, datetime(2024, 1, 2)))
        self.assertTrue(f.allows(_signal(SignalAction.SHORT)))
        self.assertFalse(f.allows(_signal(SignalAction.LONG)))


class TestTrendFilterStats(unittest.TestCase):
    def test_stats_track_vetoes_and_passes(self):
        f = TrendFilter(period=3)
        for i, p in enumerate([1.00, 1.01, 1.02]):
            f.on_bar(_bar(p, datetime(2024, 1, 1) + timedelta(hours=i)))
        f.on_bar(_bar(1.10, datetime(2024, 1, 2)))

        f.allows(_signal(SignalAction.LONG))
        f.allows(_signal(SignalAction.SHORT))
        f.allows(_signal(SignalAction.LONG))
        stats = f.stats()
        self.assertEqual(stats["passed"], 2)
        self.assertEqual(stats["vetoed_shorts"], 1)


class TestStrategyWithFilter(unittest.TestCase):
    def test_filter_vetoes_reach_strategy_output(self):
        from bossfx.strategies.sma_crossover import SMACrossoverStrategy

        class AlwaysVeto:
            period = 0

            def on_bar(self, bar):
                pass

            def allows(self, signal):
                return False

        strat = SMACrossoverStrategy(
            fast_period=2,
            slow_period=4,
            filters=[AlwaysVeto()],
        )
        closes = [1.0, 1.0, 1.0, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
        signals = []
        for i, c in enumerate(closes):
            bar = _bar(c, datetime(2024, 1, 1) + timedelta(hours=i))
            s = strat.on_bar(bar)
            if s is not None:
                signals.append(s)
        self.assertEqual(signals, [])

    def test_no_filter_preserves_original_behavior(self):
        from bossfx.strategies.sma_crossover import SMACrossoverStrategy

        strat = SMACrossoverStrategy(fast_period=2, slow_period=4)
        closes = [1.0, 1.0, 1.0, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
        signals = []
        for i, c in enumerate(closes):
            bar = _bar(c, datetime(2024, 1, 1) + timedelta(hours=i))
            s = strat.on_bar(bar)
            if s is not None:
                signals.append(s)
        self.assertTrue(any(s.action == SignalAction.LONG for s in signals))


if __name__ == "__main__":
    unittest.main()
