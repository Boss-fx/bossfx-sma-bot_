"""
scripts.run_backtest
====================

Phase 1 CLI entry point. Usage:

    python -m scripts.run_backtest --config configs/eurusd_sma_default.yaml

This is thin on purpose — all behavior is driven by the YAML config.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make package importable when running as a script
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


def main() -> int:
    parser = argparse.ArgumentParser(description="BossFx backtest runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.log_level)
    log = get_logger("bossfx.runner")
    log.info(f"Loaded config: {args.config}")

    feed = build_feed(cfg)
    strategy = SMACrossoverStrategy(
        fast_period=cfg.strategy.fast_period,
        slow_period=cfg.strategy.slow_period,
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
