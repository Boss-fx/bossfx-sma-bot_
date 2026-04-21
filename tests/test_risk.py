"""
Tests for PercentRiskManager.

The core sizing equation:
    quantity = (equity * risk_per_trade_pct) / abs(entry - stop)

Plus: drawdown circuit breaker must halt trading when DD exceeds the cap.
"""
from __future__ import annotations

import unittest
from datetime import datetime

from bossfx.core.events import BarEvent, OrderSide, SignalAction, SignalEvent
from bossfx.core.portfolio import CashPortfolio
from bossfx.risk.risk_manager import PercentRiskManager


def _bar(close):
    return BarEvent(
        symbol="EURUSD", timestamp=datetime(2024, 1, 1),
        open=close, high=close + 0.001, low=close - 0.001, close=close, volume=100.0,
    )


def _signal(action):
    return SignalEvent(
        symbol="EURUSD", timestamp=datetime(2024, 1, 1),
        action=action, strategy_id="test", reference_price=1.10,
    )


class TestPositionSizing(unittest.TestCase):
    def test_sizing_follows_the_formula(self):
        p = CashPortfolio(initial_cash=10_000)
        rm = PercentRiskManager(
            portfolio=p, risk_per_trade_pct=0.01,
            stop_loss_pct=0.01, take_profit_pct=0.02,
            max_position_pct=1.0,  # disable cap for this test
        )
        order = rm.size_order(_signal(SignalAction.LONG), _bar(1.10), equity=10_000)
        self.assertIsNotNone(order)
        # Risk cash = 100. Stop distance = 1.10 * 0.01 = 0.011. Qty = 100 / 0.011.
        expected_qty = 100 / (1.10 * 0.01)
        self.assertAlmostEqual(order.quantity, expected_qty, places=2)
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertAlmostEqual(order.stop_loss, 1.10 * 0.99, places=5)
        self.assertAlmostEqual(order.take_profit, 1.10 * 1.02, places=5)

    def test_short_sizing_and_stop_placement(self):
        p = CashPortfolio(initial_cash=10_000)
        rm = PercentRiskManager(
            portfolio=p, risk_per_trade_pct=0.01,
            stop_loss_pct=0.01, take_profit_pct=0.02, max_position_pct=1.0,
        )
        order = rm.size_order(_signal(SignalAction.SHORT), _bar(1.10), equity=10_000)
        self.assertIsNotNone(order)
        self.assertEqual(order.side, OrderSide.SELL)
        # Short: stop above, target below
        self.assertGreater(order.stop_loss, 1.10)
        self.assertLess(order.take_profit, 1.10)

    def test_max_position_cap_is_enforced(self):
        p = CashPortfolio(initial_cash=10_000)
        rm = PercentRiskManager(
            portfolio=p, risk_per_trade_pct=0.05,   # aggressive
            stop_loss_pct=0.001,                    # tiny stop => huge qty without cap
            take_profit_pct=0.002, max_position_pct=0.20,
        )
        order = rm.size_order(_signal(SignalAction.LONG), _bar(1.10), equity=10_000)
        self.assertIsNotNone(order)
        # With cap: notional ≤ 20% of equity = $2000. Quantity × price ≤ 2000.
        self.assertLessEqual(order.quantity * 1.10, 10_000 * 0.20 + 1e-6)


class TestDrawdownCircuitBreaker(unittest.TestCase):
    def test_halts_when_drawdown_exceeds_cap(self):
        p = CashPortfolio(initial_cash=10_000)
        rm = PercentRiskManager(portfolio=p, max_drawdown_pct=0.10)
        # First call establishes peak at 10k
        self.assertIsNotNone(rm.size_order(_signal(SignalAction.LONG), _bar(1.10), 10_000))
        # Simulate 15% drawdown — should be vetoed
        self.assertIsNone(rm.size_order(_signal(SignalAction.LONG), _bar(1.10), 8_500))


class TestExitHandling(unittest.TestCase):
    def test_exit_signal_with_no_position_returns_none(self):
        p = CashPortfolio(initial_cash=10_000)
        rm = PercentRiskManager(portfolio=p)
        self.assertIsNone(
            rm.size_order(_signal(SignalAction.EXIT), _bar(1.10), 10_000)
        )


if __name__ == "__main__":
    unittest.main()
