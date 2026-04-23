"""
scripts.run_backtest
====================

BossFx CLI. Usage:

    python3 -m scripts.run_backtest --config configs/eurusd_sma_default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bossfx.analytics.metrics import compute_report
from bossfx.backtest.engine import BacktestEngine
from bossfx.backtest.execution_sim import SimulatedExecutor
from bossfx.config.settings import load_config
from bossfx.core.portfolio import CashPortfolio
from bossfx.data.csv_feed import CSVDataFeed
from bossfx.risk.risk_manager import PercentRiskManager
from bossfx.strategies.sma_crossover import SMACrossoverStrategy
from bossfx.utils.logger import configure_logging, get_logger


def build_feed(cfg):
    if cfg.data.source == "csv":
        return CSVDataFeed(
            path=cfg.data.csv_path,
            symbol=cfg.data.symbol,
            timeframe=cfg.data.timeframe,
        )
    if cfg.data.source == "yfinance":
        from bossfx.data.yfinance_feed import YFinanceDataFeed

        return YFinanceDataFeed(
            symbol=cfg.data.symbol,
            start=cfg.data.start,
            end=cfg.data.end,
            interval=cfg.data.timeframe,
        )
    raise ValueError(f"Unknown data source: {cfg.data.source}")


def build_filters(cfg, log):
    """Construct the filter chain from config. Order matters for logging only."""
    filters = []
    if cfg.strategy.use_trend_filter:
        from bossfx.strategies.filters.trend import TrendFilter

        filters.append(TrendFilter(period=cfg.strategy.trend_filter_period))
        log.info(f"Trend filter ENABLED with EMA({cfg.strategy.trend_filter_period})")
    if cfg.strategy.use_volatility_filter:
        from bossfx.strategies.filters.volatility import ATRVolatilityFilter

        filters.append(
            ATRVolatilityFilter(
                atr_period=cfg.strategy.volatility_atr_period,
                lookback=cfg.strategy.volatility_lookback,
                min_ratio=cfg.strategy.volatility_min_ratio,
                max_ratio=cfg.strategy.volatility_max_ratio,
            )
        )
        log.info(
            f"Volatility filter ENABLED: ATR({cfg.strategy.volatility_atr_period}) "
            f"vs SMA({cfg.strategy.volatility_lookback}), "
            f"ratio range [{cfg.strategy.volatility_min_ratio}, "
            f"{cfg.strategy.volatility_max_ratio}]"
        )
    return filters


def print_filter_stats(filters, log):
    """Per-filter veto breakdown so we can diagnose which filter did what."""
    if not filters:
        return
    log.info("=" * 60)
    log.info("FILTER STATS (per-filter breakdown)")
    log.info("=" * 60)
    for f in filters:
        name = type(f).__name__
        stats = f.stats()
        log.info(f"  {name}:")
        for k, v in stats.items():
            if isinstance(v, float):
                log.info(f"    {k:20s} = {v:.3f}")
            else:
                log.info(f"    {k:20s} = {v}")


def main() -> int:
    parser = argparse.ArgumentParser(description="BossFx backtest runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.log_level)
    log = get_logger("bossfx.runner")
    log.info(f"Loaded config: {args.config}")

    feed = build_feed(cfg)
    filters = build_filters(cfg, log)

    strategy = SMACrossoverStrategy(
        fast_period=cfg.strategy.fast_period,
        slow_period=cfg.strategy.slow_period,
        filters=filters,
    )
    portfolio = CashPortfolio(initial_cash=cfg.risk.initial_cash)
    risk = PercentRiskManager(
        portfolio=portfolio,
        risk_per_trade_pct=cfg.risk.risk_per_trade_pct,
        stop_loss_pct=cfg.risk.stop_loss_pct,
        take_profit_pct=cfg.risk.take_profit_pct,
        max_position_pct=cfg.risk.max_position_pct,
    )
    executor = SimulatedExecutor(
        spread_pips=cfg.execution.spread_pips,
        slippage_pips=cfg.execution.slippage_pips,
        commission_per_lot=cfg.execution.commission_per_lot,
        pip_size=cfg.execution.pip_size,
        contract_size=cfg.execution.contract_size,
    )

    engine = BacktestEngine(
        data_feed=feed,
        strategy=strategy,
        risk_manager=risk,
        executor=executor,
        portfolio=portfolio,
    )
    result = engine.run()

    print_filter_stats(filters, log)

    report = compute_report(
        equity_curve=portfolio.equity_curve(),
        trades=result.trades,
        initial_cash=portfolio.initial_cash,
        timeframe=cfg.data.timeframe,
    )
    print(report.pretty())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
