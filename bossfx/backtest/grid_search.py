"""
bossfx.backtest.grid_search
===========================

Generic parameter grid search. Given a data feed and a grid of params,
run a backtest for each combination, score it, rank the results.
This is a reusable building block - walk-forward uses it, any future
optimizer can use it too.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from bossfx.analytics.metrics import PerformanceReport, compute_report
from bossfx.backtest.engine import BacktestEngine
from bossfx.backtest.execution_sim import SimulatedExecutor
from bossfx.core.interfaces import DataFeed
from bossfx.core.portfolio import CashPortfolio
from bossfx.risk.risk_manager import PercentRiskManager
from bossfx.strategies.filters.trend import TrendFilter
from bossfx.strategies.sma_crossover import SMACrossoverStrategy
from bossfx.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class GridResult:
    """One row of a grid search: parameters tried + performance achieved."""

    params: Dict[str, Any]
    report: Optional[PerformanceReport]
    num_trades: int
    score: float

    def __repr__(self) -> str:
        ret = self.report.total_return_pct if self.report else 0.0
        return (
            f"GridResult(params={self.params}, score={self.score:.3f}, "
            f"trades={self.num_trades}, return={ret:+.2f}%)"
        )


def sharpe_score(report: PerformanceReport, num_trades: int) -> float:
    """
    Default scoring function: Sharpe with a minimum-trades floor.

    Why the floor? A strategy with 1 lucky trade can post a Sharpe of 10 -
    meaningless. We require >= 5 trades before trusting the Sharpe.
    Below that, score as -inf so this combo never wins.
    """
    MIN_TRADES = 5
    if num_trades < MIN_TRADES:
        return float("-inf")
    return report.sharpe


def _build_strategy(params: Dict[str, Any]) -> SMACrossoverStrategy:
    """Build a strategy (with optional trend filter) from a params dict."""
    filters = []
    if params.get("use_trend_filter", False):
        filters.append(TrendFilter(period=params.get("trend_filter_period", 200)))
    return SMACrossoverStrategy(
        fast_period=params["fast_period"],
        slow_period=params["slow_period"],
        filters=filters,
    )


def run_single(
    data_feed: DataFeed,
    params: Dict[str, Any],
    initial_cash: float = 10_000.0,
    risk_per_trade_pct: float = 0.01,
    stop_loss_pct: float = 0.010,
    take_profit_pct: float = 0.020,
    timeframe: str = "1d",
    scorer: Callable[[PerformanceReport, int], float] = sharpe_score,
) -> GridResult:
    """Run one backtest with the given params. Returns a GridResult."""
    portfolio = CashPortfolio(initial_cash=initial_cash)
    strategy = _build_strategy(params)
    risk = PercentRiskManager(
        portfolio=portfolio,
        risk_per_trade_pct=risk_per_trade_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )
    executor = SimulatedExecutor()
    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        risk_manager=risk,
        executor=executor,
        portfolio=portfolio,
    )
    result = engine.run()

    if not portfolio.equity_curve():
        return GridResult(params=params, report=None, num_trades=0, score=float("-inf"))

    report = compute_report(
        equity_curve=portfolio.equity_curve(),
        trades=result.trades,
        initial_cash=portfolio.initial_cash,
        timeframe=timeframe,
    )
    score = scorer(report, len(result.trades))
    return GridResult(
        params=params,
        report=report,
        num_trades=len(result.trades),
        score=score,
    )


def expand_grid(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product: dict-of-lists -> list-of-dicts."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos: List[Dict[str, Any]] = []
    for bundle in itertools.product(*values):
        combos.append(dict(zip(keys, bundle)))
    return combos


def search(
    feed_factory: Callable[[], DataFeed],
    grid: Dict[str, Sequence[Any]],
    scorer: Callable[[PerformanceReport, int], float] = sharpe_score,
    timeframe: str = "1d",
    **backtest_kwargs: Any,
) -> List[GridResult]:
    """
    Run a full grid search. Returns results sorted by score descending.

    Why feed_factory instead of feed? Feeds are stateful (the stream
    iterator advances). We need a fresh feed per combo. The factory is
    how we get clean feeds repeatedly.
    """
    combos = expand_grid(grid)
    log.info(f"Grid search: {len(combos)} combinations to evaluate")
    results: List[GridResult] = []

    for i, params in enumerate(combos, 1):
        # Skip invalid SMA combos (fast >= slow) silently
        if "fast_period" in params and "slow_period" in params:
            if params["fast_period"] >= params["slow_period"]:
                continue
        try:
            feed = feed_factory()
            res = run_single(
                data_feed=feed,
                params=params,
                scorer=scorer,
                timeframe=timeframe,
                **backtest_kwargs,
            )
            results.append(res)
        except Exception as e:
            log.warning(f"  [{i}/{len(combos)}] {params} FAILED: {e}")

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def best_of(results: Sequence[GridResult]) -> Optional[GridResult]:
    """Top-scoring credible result, or None if nothing cleared the trade floor."""
    for r in results:
        if r.score != float("-inf"):
            return r
    return None
