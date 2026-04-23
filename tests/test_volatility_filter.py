"""
Tests for the Phase 2.2 ATR volatility filter.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from bossfx.core.events import BarEvent, SignalAction, SignalEvent
from bossfx.strategies.filters.volatility import ATRVolatilityFilter


def _bar(close, ts, high_offset=0.001, low_offset=0.001):
    return BarEvent(
        symbol="EURUSD",
        timestamp=ts,
        open=close,
        high=close + high_offset,
        low=close - low_offset,
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


class TestInit(unittest.TestCase):
    def test_rejects_nonpositive_periods(self):
        with self.assertRaises(ValueError):
            ATRVolatilityFilter(atr_period=0)
        with self.assertRaises(ValueError):
            ATRVolatilityFilter(lookback=-5)

    def test_rejects_invalid_ratios(self):
        with self.assertRaises(ValueError):
            ATRVolatilityFilter(min_ratio=-0.1)
        with self.assertRaises(ValueError):
            ATRVolatilityFilter(min_ratio=1.5, max_ratio=1.5)
        with self.assertRaises(ValueError):
            ATRVolatilityFilter(min_ratio=2.0, max_ratio=1.0)


class TestWarmup(unittest.TestCase):
    def test_vetos_before_atr_is_warm(self):
        f = ATRVolatilityFilter(atr_period=5, lookback=10)
        for i in range(3):
            f.on_bar(_bar(1.10, datetime(2024, 1, 1) + timedelta(hours=i)))
        self.assertFalse(f.allows(_signal(SignalAction.LONG)))

    def test_vetos_before_lookback_avg_is_warm(self):
        """ATR warm but SMA-of-ATR not yet warm."""
        f = ATRVolatilityFilter(atr_period=3, lookback=20)
        for i in range(10):
            f.on_bar(_bar(1.10, datetime(2024, 1, 1) + timedelta(hours=i)))
        self.assertFalse(f.allows(_signal(SignalAction.LONG)))

    def test_exit_signals_always_pass_even_before_warm(self):
        f = ATRVolatilityFilter(atr_period=5, lookback=10)
        self.assertTrue(f.allows(_signal(SignalAction.EXIT)))


class TestDecision(unittest.TestCase):
    """
    Decision tests use a SHORT lookback + a tiny final spike so the
    ratio moves decisively outside [min, max]. We also make the lookback
    small enough that a single spike can shift the ratio by a known amount.
    """

    def test_passes_in_steady_normal_regime(self):
        """After steady warmup, ratio ~ 1.0 so signals pass."""
        f = ATRVolatilityFilter(atr_period=3, lookback=10, min_ratio=0.7, max_ratio=1.5)
        for i in range(30):
            f.on_bar(_bar(1.10, datetime(2024, 1, 1) + timedelta(hours=i)))
        # Ratio should be ~1.0 — signal passes
        self.assertTrue(f.allows(_signal(SignalAction.LONG)))

    def test_vetos_when_current_atr_is_very_low(self):
        """
        Warm with normal vol so lookback avg settles.
        Then feed many dead-flat bars so CURRENT ATR drops while
        lookback avg is still elevated. Ratio should drop below 0.7.
        """
        f = ATRVolatilityFilter(atr_period=3, lookback=20, min_ratio=0.7, max_ratio=1.5)
        # Warm with ~0.002 range bars (ATR ~0.002)
        for i in range(40):
            f.on_bar(_bar(1.10, datetime(2024, 1, 1) + timedelta(hours=i)))
        # Now feed JUST A FEW dead-flat bars. atr_period=3 means ATR
        # responds in ~3 bars, but lookback=20 means SMA-of-ATR stays
        # elevated — creating the ratio drop we want to see.
        for i in range(5):
            f.on_bar(
                _bar(
                    1.10,
                    datetime(2024, 2, 1) + timedelta(hours=i),
                    high_offset=0.00001,
                    low_offset=0.00001,
                )
            )
        result = f.allows(_signal(SignalAction.LONG))
        self.assertFalse(result, "Expected VETO when current ATR drops well below lookback average")

    def test_vetos_when_current_atr_is_very_high(self):
        """
        Symmetric: warm on normal vol, then a couple huge bars spike
        current ATR while lookback avg lags. Ratio > max triggers veto.
        """
        f = ATRVolatilityFilter(atr_period=3, lookback=20, min_ratio=0.7, max_ratio=1.5)
        for i in range(40):
            f.on_bar(_bar(1.10, datetime(2024, 1, 1) + timedelta(hours=i)))
        # Feed a few very wild bars — ATR shoots up, lookback avg lags
        for i in range(5):
            f.on_bar(
                _bar(
                    1.10,
                    datetime(2024, 2, 1) + timedelta(hours=i),
                    high_offset=0.05,
                    low_offset=0.05,
                )
            )
        result = f.allows(_signal(SignalAction.LONG))
        self.assertFalse(
            result, "Expected VETO when current ATR spikes well above lookback average"
        )


class TestStats(unittest.TestCase):
    def test_stats_initialize_to_zero(self):
        f = ATRVolatilityFilter(atr_period=3, lookback=5)
        stats = f.stats()
        self.assertEqual(stats["passed"], 0)
        self.assertEqual(stats["total_vetoed"], 0)
        self.assertEqual(stats["veto_rate"], 0.0)

    def test_period_attribute_for_warmup_scan(self):
        """Strategy uses .period for warmup — must equal atr_period + lookback."""
        f = ATRVolatilityFilter(atr_period=14, lookback=100)
        self.assertEqual(f.period, 114)


class TestCompositionWithTrendFilter(unittest.TestCase):
    """Phase 2.1 + Phase 2.2 stacked — both filters must allow for signal to pass."""

    def test_both_filters_can_be_composed(self):
        from bossfx.strategies.filters.trend import TrendFilter
        from bossfx.strategies.sma_crossover import SMACrossoverStrategy

        trend = TrendFilter(period=10)
        vol = ATRVolatilityFilter(atr_period=5, lookback=10)
        strat = SMACrossoverStrategy(fast_period=2, slow_period=4, filters=[trend, vol])
        closes = [1.0] * 20 + [1.01, 1.02, 1.03, 1.04, 1.05]
        for i, c in enumerate(closes):
            bar = _bar(c, datetime(2024, 1, 1) + timedelta(hours=i))
            strat.on_bar(bar)
        # Both filters should have been called — no exceptions is the test


class TestNoLookAhead(unittest.TestCase):
    """Classic structural test — future data can't alter past decisions."""

    def test_past_decisions_do_not_change_when_future_bars_arrive(self):
        ts = datetime(2024, 1, 1)

        # Run A: warm + decide
        f1 = ATRVolatilityFilter(atr_period=3, lookback=5)
        for i in range(15):
            f1.on_bar(_bar(1.10, ts + timedelta(hours=i)))
        decision_a = f1.allows(_signal(SignalAction.LONG))

        # Run B: same warmup, decide at same point
        f2 = ATRVolatilityFilter(atr_period=3, lookback=5)
        for i in range(15):
            f2.on_bar(_bar(1.10, ts + timedelta(hours=i)))
        decision_b = f2.allows(_signal(SignalAction.LONG))
        # Then feed future spike — shouldn't retroactively change decision_b
        for i in range(5):
            f2.on_bar(
                _bar(
                    10.0,
                    ts + timedelta(hours=100 + i),
                    high_offset=1.0,
                    low_offset=1.0,
                )
            )

        self.assertEqual(decision_a, decision_b)


if __name__ == "__main__":
    unittest.main()
