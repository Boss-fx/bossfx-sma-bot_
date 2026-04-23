"""
scripts.run_walkforward
=======================

CLI for rolling walk-forward validation. Usage:

    python3 -m scripts.run_walkforward --config configs/walkforward_5y.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from bossfx.backtest.walkforward import run_walkforward
from bossfx.config.settings import load_config
from bossfx.data.csv_feed import CSVDataFeed
from bossfx.utils.logger import configure_logging, get_logger


def build_feed_factory(cfg):
    """Return a callable that produces fresh feeds on demand."""
    if cfg.data.source == "csv":

        def factory():
            return CSVDataFeed(
                path=cfg.data.csv_path,
                symbol=cfg.data.symbol,
                timeframe=cfg.data.timeframe,
            )

        return factory
    if cfg.data.source == "yfinance":
        from bossfx.data.yfinance_feed import YFinanceDataFeed

        def factory():
            return YFinanceDataFeed(
                symbol=cfg.data.symbol,
                start=cfg.data.start,
                end=cfg.data.end,
                interval=cfg.data.timeframe,
            )

        return factory
    raise ValueError(f"Unknown data source: {cfg.data.source}")


def main() -> int:
    parser = argparse.ArgumentParser(description="BossFx walk-forward runner")
    parser.add_argument("--config", required=True, help="Path to walk-forward YAML")
    args = parser.parse_args()

    # Load raw YAML to get walk-forward-specific fields + the full config
    raw = yaml.safe_load(Path(args.config).read_text())
    wf_cfg = raw.get("walkforward", {})

    cfg = load_config(args.config)
    configure_logging(cfg.log_level)
    log = get_logger("bossfx.walkforward_runner")
    log.info(f"Loaded config: {args.config}")

    feed_factory = build_feed_factory(cfg)

    grid = {
        "fast_period": wf_cfg.get("fast_periods", [10, 15, 20, 25, 30]),
        "slow_period": wf_cfg.get("slow_periods", [40, 50, 75, 100]),
        "use_trend_filter": wf_cfg.get("use_trend_filter_options", [False, True]),
        "trend_filter_period": wf_cfg.get("trend_filter_periods", [200]),
    }

    log.info(f"Grid: {grid}")

    report = run_walkforward(
        feed_factory=feed_factory,
        grid=grid,
        start=cfg.data.start,
        end=cfg.data.end,
        train_months=wf_cfg.get("train_months", 24),
        test_months=wf_cfg.get("test_months", 6),
        step_months=wf_cfg.get("step_months", 6),
        timeframe=cfg.data.timeframe,
        initial_cash=cfg.risk.initial_cash,
        risk_per_trade_pct=cfg.risk.risk_per_trade_pct,
        stop_loss_pct=cfg.risk.stop_loss_pct,
        take_profit_pct=cfg.risk.take_profit_pct,
    )

    print(report.pretty())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
