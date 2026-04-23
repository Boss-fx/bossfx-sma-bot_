"""
bossfx.backtest.walkforward
===========================

Rolling walk-forward validation.

For each window:
  1. Slice the data into train (e.g. 2 years) and test (e.g. 6 months).
  2. Grid-search on train to find the best parameters.
  3. Evaluate THOSE parameters - and only those - on the test slice.
  4. Slide the window forward and repeat.

The concatenated out-of-sample test performance is the honest estimate
of what the strategy would have produced live.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from bossfx.backtest.grid_search import (
    GridResult,
    best_of,
    run_single,
    search,
    sharpe_score,
)
from bossfx.core.interfaces import DataFeed
from bossfx.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Window:
    """One rolling window: train range, test range, winner, and OOS score."""

    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    winner: Optional[GridResult] = None
    oos_result: Optional[GridResult] = None

    def summary(self) -> str:
        ts = self.train_start.strftime("%Y-%m-%d")
        te = self.train_end.strftime("%Y-%m-%d")
        vs = self.test_start.strftime("%Y-%m-%d")
        ve = self.test_end.strftime("%Y-%m-%d")
        if self.winner is None:
            return f"[{self.index}] train {ts}->{te} test {vs}->{ve} | NO WINNER"
        w = self.winner
        if self.oos_result is None or self.oos_result.report is None:
            return (
                f"[{self.index}] train {ts}->{te} test {vs}->{ve} | "
                f"IS {w.params} Sharpe={w.score:.2f} | OOS: no trades"
            )
        oos = self.oos_result
        return (
            f"[{self.index}] train {ts}->{te} test {vs}->{ve} | "
            f"IS {w.params} IS_Sharpe={w.score:.2f} | "
            f"OOS_Sharpe={oos.score:.2f} ret={oos.report.total_return_pct:+.2f}% "
            f"tr={oos.num_trades}"
        )


@dataclass
class WalkForwardReport:
    """Aggregates all windows into a single honest report."""

    windows: List[Window] = field(default_factory=list)

    @property
    def num_windows(self) -> int:
        return len([w for w in self.windows if w.oos_result is not None])

    @property
    def oos_sharpes(self) -> List[float]:
        return [
            w.oos_result.score
            for w in self.windows
            if w.oos_result is not None and w.oos_result.score != float("-inf")
        ]

    @property
    def oos_returns_pct(self) -> List[float]:
        return [
            w.oos_result.report.total_return_pct
            for w in self.windows
            if w.oos_result is not None and w.oos_result.report is not None
        ]

    @property
    def avg_oos_sharpe(self) -> float:
        vals = self.oos_sharpes
        return statistics.mean(vals) if vals else 0.0

    @property
    def median_oos_sharpe(self) -> float:
        vals = self.oos_sharpes
        return statistics.median(vals) if vals else 0.0

    @property
    def total_oos_return_pct(self) -> float:
        """Compounded return across all OOS test slices."""
        pct = 1.0
        for r in self.oos_returns_pct:
            pct *= 1 + (r / 100)
        return (pct - 1) * 100

    def pretty(self) -> str:
        lines = [
            "",
            "=" * 66,
            "              BossFx Walk-Forward Report",
            "=" * 66,
            f"  Windows evaluated     : {self.num_windows}",
            f"  Avg OOS Sharpe        : {self.avg_oos_sharpe:+.3f}",
            f"  Median OOS Sharpe     : {self.median_oos_sharpe:+.3f}",
            f"  Compounded OOS return : {self.total_oos_return_pct:+.2f}%",
            "-" * 66,
            "  Per-window detail:",
        ]
        for w in self.windows:
            lines.append(f"  {w.summary()}")
        lines.append("=" * 66)
        return "\n".join(lines)


def build_windows(
    start: datetime,
    end: datetime,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
) -> List[Tuple[datetime, datetime, datetime, datetime]]:
    """
    Generate (train_start, train_end, test_start, test_end) tuples.
    Uses calendar-accurate month math.
    """
    windows: List[Tuple[datetime, datetime, datetime, datetime]] = []
    train_start = start
    while True:
        train_end_ts = pd.Timestamp(train_start) + pd.DateOffset(months=train_months)
        test_start_ts = train_end_ts
        test_end_ts = test_start_ts + pd.DateOffset(months=test_months)
        if test_end_ts > pd.Timestamp(end) + pd.Timedelta(days=1):
            break
        windows.append(
            (
                train_start,
                train_end_ts.to_pydatetime(),
                test_start_ts.to_pydatetime(),
                test_end_ts.to_pydatetime(),
            )
        )
        next_start_ts = pd.Timestamp(train_start) + pd.DateOffset(months=step_months)
        train_start = next_start_ts.to_pydatetime()
    return windows


def _filter_feed_factory(
    full_feed_factory: Callable[[], DataFeed],
    start: datetime,
    end: datetime,
) -> Callable[[], DataFeed]:
    """
    Wrap a feed factory so only bars in [start, end) are streamed.
    This is how we enforce the train/test boundary — the feed itself
    refuses to yield bars outside the window.
    """

    class _Slice:
        def __init__(self) -> None:
            self._inner = full_feed_factory()

        @property
        def symbol(self) -> str:
            return self._inner.symbol

        @property
        def timeframe(self) -> str:
            return self._inner.timeframe

        def stream(self):
            # Make naive/aware datetimes comparable
            s = (
                pd.Timestamp(start).tz_localize("UTC")
                if pd.Timestamp(start).tz is None
                else pd.Timestamp(start)
            )
            e = (
                pd.Timestamp(end).tz_localize("UTC")
                if pd.Timestamp(end).tz is None
                else pd.Timestamp(end)
            )
            for bar in self._inner.stream():
                bar_ts = pd.Timestamp(bar.timestamp)
                if bar_ts.tz is None:
                    bar_ts = bar_ts.tz_localize("UTC")
                if s <= bar_ts < e:
                    yield bar

    return _Slice


def run_walkforward(
    feed_factory: Callable[[], DataFeed],
    grid: Dict[str, Sequence[Any]],
    start: datetime,
    end: datetime,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
    timeframe: str = "1d",
    scorer: Callable = sharpe_score,
    **backtest_kwargs: Any,
) -> WalkForwardReport:
    """Run rolling walk-forward and return the aggregated honest report."""
    window_ranges = build_windows(
        start=start,
        end=end,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
    )
    log.info(f"Walk-forward: {len(window_ranges)} windows")

    report = WalkForwardReport()

    for i, (train_start, train_end, test_start, test_end) in enumerate(window_ranges):
        log.info(f"--- Window {i + 1}/{len(window_ranges)} ---")
        log.info(f"    Train: {train_start.date()} -> {train_end.date()}")
        log.info(f"    Test:  {test_start.date()} -> {test_end.date()}")

        train_factory = _filter_feed_factory(feed_factory, train_start, train_end)
        test_factory = _filter_feed_factory(feed_factory, test_start, test_end)

        results = search(
            feed_factory=train_factory,
            grid=grid,
            scorer=scorer,
            timeframe=timeframe,
            **backtest_kwargs,
        )
        winner = best_of(results)
        window = Window(
            index=i,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            winner=winner,
        )

        if winner is None:
            log.warning(f"    No credible winner on train window {i + 1}")
            report.windows.append(window)
            continue

        log.info(f"    IS winner: {winner.params} Sharpe={winner.score:.3f}")

        test_feed = test_factory()
        oos = run_single(
            data_feed=test_feed,
            params=winner.params,
            scorer=scorer,
            timeframe=timeframe,
            **backtest_kwargs,
        )
        window.oos_result = oos
        oos_ret = oos.report.total_return_pct if oos.report else 0.0
        log.info(f"    OOS Sharpe={oos.score:.3f} trades={oos.num_trades} ret={oos_ret:+.2f}%")
        report.windows.append(window)

    return report
