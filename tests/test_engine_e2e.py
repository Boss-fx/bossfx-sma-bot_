"""
End-to-end engine test.

This test wires up every real component (feed, strategy, risk, executor,
portfolio) against our deterministic fixture CSV and verifies:

  * The engine runs to completion.
  * The equity curve has one point per bar.
  * No negative equity (would mean catastrophic accounting bug).
  * Trades are well-formed (have entry + exit).
"""

from __future__ import annotations

import unittest
from pathlib import Path

from bossfx.backtest.engine import BacktestEngine
from bossfx.backtest.execution_sim import SimulatedExecutor
from bossfx.core.portfolio import CashPortfolio
from bossfx.data.csv_feed import CSVDataFeed
from bossfx.risk.risk_manager import PercentRiskManager
from bossfx.strategies.sma_crossover import SMACrossoverStrategy


FIXTURE = Path(__file__).parent / "fixtures" / "eurusd_1h_sample.csv"


class TestEngineEndToEnd(unittest.TestCase):
    def test_full_pipeline_runs(self):
        feed = CSVDataFeed(path=FIXTURE, symbol="EURUSD", timeframe="1h")
        strategy = SMACrossoverStrategy(fast_period=20, slow_period=50)
        portfolio = CashPortfolio(initial_cash=10_000.0)
        risk = PercentRiskManager(
            portfolio=portfolio,
            risk_per_trade_pct=0.01,
            stop_loss_pct=0.005,
            take_profit_pct=0.010,
            max_position_pct=0.20,
        )
        executor = SimulatedExecutor(
            spread_pips=1.0,
            slippage_pips=0.5,
            commission_per_lot=7.0,
        )
        engine = BacktestEngine(
            data_feed=feed,
            strategy=strategy,
            risk_manager=risk,
            executor=executor,
            portfolio=portfolio,
        )

        result = engine.run()

        # Ran through all 2000 bars
        self.assertEqual(result.bars_processed, 2000)
        # One equity point per bar
        self.assertEqual(len(portfolio.equity_curve()), 2000)
        # No negative equity — catastrophic if this fails
        min_equity = min(e for _, e in portfolio.equity_curve())
        self.assertGreater(min_equity, 0, "Negative equity indicates accounting bug")
        # Some trades should have fired on 2000 bars with these SMA params
        self.assertGreater(result.fills_executed, 0)

    def test_trade_log_entries_are_well_formed(self):
        feed = CSVDataFeed(path=FIXTURE, symbol="EURUSD", timeframe="1h")
        strategy = SMACrossoverStrategy(fast_period=20, slow_period=50)
        portfolio = CashPortfolio(initial_cash=10_000.0)
        risk = PercentRiskManager(portfolio=portfolio)
        executor = SimulatedExecutor()
        engine = BacktestEngine(
            data_feed=feed,
            strategy=strategy,
            risk_manager=risk,
            executor=executor,
            portfolio=portfolio,
        )
        result = engine.run()
        for t in result.trades:
            self.assertIn("entry_time", t)
            self.assertIn("exit_time", t)
            self.assertIn("entry_price", t)
            self.assertIn("exit_price", t)
            self.assertIn("gross_pnl", t)
            self.assertGreater(t["entry_price"], 0)
            self.assertGreater(t["exit_price"], 0)
            self.assertGreaterEqual(t["exit_time"], t["entry_time"])


if __name__ == "__main__":
    unittest.main()
