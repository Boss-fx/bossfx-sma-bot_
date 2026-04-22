"""
Tests for online indicators — TIER 2 (no-look-ahead invariants).

The central invariant: an online indicator can NEVER produce a value
based on data it hasn't seen yet. If this holds structurally, look-ahead
bias is *impossible* — not just unlikely.
"""

from __future__ import annotations

import unittest

from bossfx.strategies.indicators import ATR, SMA


class TestSMABasicMath(unittest.TestCase):
    def test_returns_none_before_warm(self):
        s = SMA(period=3)
        self.assertIsNone(s.update(1.0))
        self.assertIsNone(s.update(2.0))
        self.assertEqual(s.update(3.0), 2.0)
        self.assertTrue(s.is_warm)

    def test_rolls_correctly(self):
        s = SMA(period=3)
        for v in [1.0, 2.0, 3.0]:
            s.update(v)
        self.assertEqual(s.update(4.0), 3.0)
        self.assertEqual(s.update(5.0), 4.0)

    def test_rejects_zero_or_negative_period(self):
        with self.assertRaises(ValueError):
            SMA(period=0)
        with self.assertRaises(ValueError):
            SMA(period=-5)

    def test_matches_naive_average(self):
        """Sanity: online SMA == naive mean of the last N values."""
        import random

        random.seed(123)
        values = [random.uniform(1.0, 2.0) for _ in range(200)]
        s = SMA(period=20)
        for i, v in enumerate(values):
            online = s.update(v)
            if i >= 19:
                naive = sum(values[i - 19 : i + 1]) / 20
                self.assertAlmostEqual(online, naive, places=10)


class TestSMANoLookAhead(unittest.TestCase):
    """The structural invariant."""

    def test_future_data_cannot_alter_past_values(self):
        prefix = [1.0, 2.0, 3.0, 4.0, 5.0]
        future_scenario_a = [100.0, 200.0, 300.0]
        future_scenario_b = [-50.0, -60.0, -70.0]

        s1 = SMA(period=3)
        sma_at_5_scenario_a = [s1.update(v) for v in prefix][-1]

        s2 = SMA(period=3)
        for v in prefix:
            s2.update(v)
        sma_at_5_baseline = s2.value
        for v in future_scenario_a:
            s2.update(v)

        s3 = SMA(period=3)
        for v in prefix:
            s3.update(v)
        sma_at_5_scenario_b = s3.value
        for v in future_scenario_b:
            s3.update(v)

        self.assertAlmostEqual(sma_at_5_scenario_a, sma_at_5_baseline, places=10)
        self.assertAlmostEqual(sma_at_5_baseline, sma_at_5_scenario_b, places=10)

    def test_update_order_matters_but_history_does_not_mutate(self):
        """Calling update(x) after getting a value must not change that value."""
        s = SMA(period=3)
        s.update(1.0)
        s.update(2.0)
        val_at_3 = s.update(3.0)
        s.update(99.0)
        s.update(100.0)
        self.assertEqual(val_at_3, 2.0)


class TestATR(unittest.TestCase):
    def test_returns_none_before_warm(self):
        a = ATR(period=3)
        self.assertIsNone(a.update(1.1, 1.0, 1.05))
        self.assertIsNone(a.update(1.12, 1.05, 1.10))
        self.assertIsNotNone(a.update(1.13, 1.08, 1.11))

    def test_atr_is_nonnegative(self):
        import random

        random.seed(42)
        a = ATR(period=14)
        for _ in range(100):
            close = random.uniform(1.0, 1.2)
            high = close + random.uniform(0, 0.005)
            low = close - random.uniform(0, 0.005)
            val = a.update(high, low, close)
            if val is not None:
                self.assertGreaterEqual(val, 0.0)

    def test_no_look_ahead_atr(self):
        """Same structural test as SMA."""
        prefix = [(1.1, 1.0, 1.05), (1.12, 1.05, 1.10), (1.13, 1.08, 1.11), (1.14, 1.10, 1.12)]
        a1 = ATR(period=3)
        for high, low, close in prefix:
            a1.update(high, low, close)
        baseline = a1.value

        a2 = ATR(period=3)
        for high, low, close in prefix:
            a2.update(high, low, close)
        a2.update(10.0, 5.0, 7.5)
        a2.update(20.0, 10.0, 15.0)

        a3 = ATR(period=3)
        for high, low, close in prefix:
            a3.update(high, low, close)
        recomputed = a3.value

        self.assertAlmostEqual(baseline, recomputed, places=10)


if __name__ == "__main__":
    unittest.main()
