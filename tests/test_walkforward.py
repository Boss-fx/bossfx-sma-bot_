"""
Tests for Phase 3 walk-forward validation framework.
Covers grid expansion, window math, and the critical anti-leak invariant.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import List

from bossfx.backtest.grid_search import expand_grid, sharpe_score
from bossfx.backtest.walkforward import (
    _filter_feed_factory,
    build_windows,
)
from bossfx.core.events import BarEvent


class TestGridExpansion(unittest.TestCase):
    def test_cartesian_product(self):
        grid = {"a": [1, 2], "b": [10, 20]}
        combos = expand_grid(grid)
        self.assertEqual(len(combos), 4)
        self.assertIn({"a": 1, "b": 10}, combos)
        self.assertIn({"a": 2, "b": 20}, combos)

    def test_single_key(self):
        combos = expand_grid({"a": [1, 2, 3]})
        self.assertEqual(combos, [{"a": 1}, {"a": 2}, {"a": 3}])

    def test_three_keys(self):
        combos = expand_grid({"a": [1, 2], "b": [3, 4], "c": [5]})
        self.assertEqual(len(combos), 4)


class TestSharpeScore(unittest.TestCase):
    def test_below_min_trades_returns_neg_inf(self):
        class FakeReport:
            sharpe = 5.0

        self.assertEqual(sharpe_score(FakeReport(), num_trades=3), float("-inf"))

    def test_above_min_trades_returns_sharpe(self):
        class FakeReport:
            sharpe = 1.5

        self.assertEqual(sharpe_score(FakeReport(), num_trades=10), 1.5)


class TestWindowBuilding(unittest.TestCase):
    def test_generates_expected_count_on_5_years(self):
        windows = build_windows(
            start=datetime(2020, 1, 1),
            end=datetime(2025, 1, 1),
            train_months=24,
            test_months=6,
            step_months=6,
        )
        # First window: train 2020-01 to 2022-01, test 2022-01 to 2022-07
        # Last window: train 2022-07 to 2024-07, test 2024-07 to 2025-01
        # So 2022, 2023, 2024 -> each year has 2 starts -> ~5 windows total
        self.assertGreaterEqual(len(windows), 4)
        self.assertLessEqual(len(windows), 6)

    def test_first_window_dates(self):
        windows = build_windows(
            start=datetime(2020, 1, 1),
            end=datetime(2025, 1, 1),
            train_months=24,
            test_months=6,
            step_months=6,
        )
        train_start, train_end, test_start, test_end = windows[0]
        self.assertEqual(train_start, datetime(2020, 1, 1))
        self.assertEqual(train_end.date(), datetime(2022, 1, 1).date())
        self.assertEqual(test_start.date(), datetime(2022, 1, 1).date())
        self.assertEqual(test_end.date(), datetime(2022, 7, 1).date())

    def test_adjacent_windows_step_by_step_months(self):
        windows = build_windows(
            start=datetime(2020, 1, 1),
            end=datetime(2025, 1, 1),
            train_months=24,
            test_months=6,
            step_months=6,
        )
        w0_train_start = windows[0][0]
        w1_train_start = windows[1][0]
        # Two starts should be ~6 months apart
        diff_days = (w1_train_start - w0_train_start).days
        self.assertTrue(170 <= diff_days <= 190)


class _FakeFeed:
    """Emits one bar per hour from 2020-01-01 for N bars."""

    def __init__(self, n_bars: int = 100) -> None:
        self._n = n_bars

    @property
    def symbol(self) -> str:
        return "EURUSD"

    @property
    def timeframe(self) -> str:
        return "1h"

    def stream(self):
        base = datetime(2020, 1, 1, tzinfo=timezone.utc)
        for i in range(self._n):
            ts = base + timedelta(hours=i)
            yield BarEvent(
                symbol="EURUSD",
                timestamp=ts,
                open=1.1,
                high=1.11,
                low=1.09,
                close=1.1,
                volume=100.0,
            )


class TestFilterFeedFactory(unittest.TestCase):
    """
    THE CRITICAL INVARIANT: the filter-wrapped feed must NEVER yield bars
    outside its [start, end) window. If it does, train data leaks into test
    (or vice versa) and walk-forward is lying.
    """

    def test_only_yields_bars_in_window(self):
        factory = _filter_feed_factory(
            full_feed_factory=lambda: _FakeFeed(n_bars=48),  # 48 hours
            start=datetime(2020, 1, 1, 10, 0, tzinfo=timezone.utc),
            end=datetime(2020, 1, 1, 20, 0, tzinfo=timezone.utc),
        )
        feed = factory()
        bars: List[BarEvent] = list(feed.stream())
        # Window is 10 hours long -> expect exactly 10 bars
        self.assertEqual(len(bars), 10)
        for b in bars:
            self.assertGreaterEqual(b.timestamp.hour, 10)
            self.assertLess(b.timestamp.hour, 20)

    def test_empty_window_yields_nothing(self):
        factory = _filter_feed_factory(
            full_feed_factory=lambda: _FakeFeed(n_bars=10),
            start=datetime(2030, 1, 1, tzinfo=timezone.utc),
            end=datetime(2030, 6, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(list(factory().stream()), [])

    def test_does_not_leak_past_end(self):
        factory = _filter_feed_factory(
            full_feed_factory=lambda: _FakeFeed(n_bars=100),
            start=datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc),
            end=datetime(2020, 1, 1, 5, 0, tzinfo=timezone.utc),
        )
        bars = list(factory().stream())
        self.assertEqual(len(bars), 5)  # hours 0,1,2,3,4


if __name__ == "__main__":
    unittest.main()
