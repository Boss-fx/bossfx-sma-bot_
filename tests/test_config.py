"""
Tests for config loading/validation.

Config mistakes are a leading cause of silent backtest lies. If someone
types fast=50, slow=20 by accident and we don't catch it, they get
garbage. These tests enforce that bad configs die at load time.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent

from bossfx.config.settings import (
    BacktestConfig, DataConfig, RiskConfig, StrategyConfig, load_config,
)


def _write_yaml(body: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(dedent(body))
    f.close()
    return Path(f.name)


class TestValidation(unittest.TestCase):
    def test_fast_must_be_less_than_slow(self):
        with self.assertRaises(ValueError):
            StrategyConfig(fast_period=50, slow_period=20).validate()

    def test_positive_cash_required(self):
        with self.assertRaises(ValueError):
            RiskConfig(initial_cash=-100).validate()

    def test_end_after_start(self):
        from datetime import datetime
        dc = DataConfig(
            source="csv", csv_path="foo.csv",
            start=datetime(2024, 1, 2), end=datetime(2024, 1, 1),
        )
        with self.assertRaises(ValueError):
            dc.validate()

    def test_csv_path_required_for_csv_source(self):
        with self.assertRaises(ValueError):
            DataConfig(source="csv", csv_path=None).validate()


class TestYAMLLoading(unittest.TestCase):
    def test_loads_valid_yaml(self):
        p = _write_yaml("""
            data:
              source: csv
              symbol: EURUSD
              timeframe: 1h
              start: 2023-01-01
              end: 2024-01-01
              csv_path: tests/fixtures/eurusd_1h_sample.csv
            strategy:
              fast_period: 20
              slow_period: 50
            risk:
              initial_cash: 10000.0
              risk_per_trade_pct: 0.01
              stop_loss_pct: 0.01
              take_profit_pct: 0.02
            execution:
              spread_pips: 1.0
              slippage_pips: 0.5
        """)
        try:
            cfg = load_config(p)
            self.assertIsInstance(cfg, BacktestConfig)
            self.assertEqual(cfg.strategy.fast_period, 20)
            self.assertEqual(cfg.data.symbol, "EURUSD")
        finally:
            os.unlink(p)

    def test_rejects_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


if __name__ == "__main__":
    unittest.main()
