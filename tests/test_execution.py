"""
Tests for SimulatedExecutor.

Invariants:
  * Buyer pays MORE than mid (spread + slippage ADDED)
  * Seller receives LESS than mid (spread + slippage SUBTRACTED)
  * Commission is always positive (cost)
  * Intrabar stop check: if bar touches stop, we fill at the stop level
"""

from __future__ import annotations

import unittest
from datetime import datetime

from bossfx.backtest.execution_sim import SimulatedExecutor
from bossfx.core.events import (
    BarEvent,
    OrderEvent,
    OrderSide,
    OrderType,
)


def _next_bar(open_price, high=None, low=None):
    high = high if high is not None else open_price + 0.005
    low = low if low is not None else open_price - 0.005
    return BarEvent(
        symbol="EURUSD",
        timestamp=datetime(2024, 1, 1, 1),
        open=open_price,
        high=high,
        low=low,
        close=open_price,
        volume=100.0,
    )


def _order(side):
    return OrderEvent(
        symbol="EURUSD",
        timestamp=datetime(2024, 1, 1),
        side=side,
        order_type=OrderType.MARKET,
        quantity=100_000,
        order_id="test",
    )


class TestExecutionCosts(unittest.TestCase):
    def test_buyer_pays_more_than_mid(self):
        exe = SimulatedExecutor(spread_pips=1.0, slippage_pips=0.5)
        fill = exe.execute(_order(OrderSide.BUY), _next_bar(1.10000))
        self.assertIsNotNone(fill)
        # Mid = 1.10000. Half spread = 0.00005. Slippage = 0.00005. Total = 0.0001
        self.assertAlmostEqual(fill.fill_price, 1.10000 + 0.00005 + 0.00005, places=6)

    def test_seller_receives_less_than_mid(self):
        exe = SimulatedExecutor(spread_pips=1.0, slippage_pips=0.5)
        fill = exe.execute(_order(OrderSide.SELL), _next_bar(1.10000))
        self.assertIsNotNone(fill)
        self.assertAlmostEqual(fill.fill_price, 1.10000 - 0.00005 - 0.00005, places=6)

    def test_commission_proportional_to_lots(self):
        exe = SimulatedExecutor(commission_per_lot=7.0, contract_size=100_000)
        # 100k qty = exactly 1 lot -> $7 commission
        fill = exe.execute(_order(OrderSide.BUY), _next_bar(1.10000))
        self.assertAlmostEqual(fill.commission, 7.0, places=4)

    def test_slippage_recorded_in_fill(self):
        exe = SimulatedExecutor(spread_pips=2.0, slippage_pips=1.0)
        fill = exe.execute(_order(OrderSide.BUY), _next_bar(1.10000))
        # slippage field is the deviation from the mid
        self.assertGreater(fill.slippage, 0)


class TestIntrabarStopCheck(unittest.TestCase):
    def test_long_stop_hit_when_low_below_stop(self):
        exe = SimulatedExecutor()
        # Long position: stop at 1.095. Bar low = 1.09 -> stop triggered.
        bar = BarEvent(
            symbol="EURUSD",
            timestamp=datetime(2024, 1, 1),
            open=1.10,
            high=1.11,
            low=1.09,
            close=1.095,
            volume=1.0,
        )
        result = exe.check_intrabar_stops(
            bar,
            OrderSide.BUY,
            stop_loss=1.095,
            take_profit=1.15,
        )
        self.assertEqual(result, ("stop", 1.095))

    def test_long_target_hit_when_high_above_target(self):
        exe = SimulatedExecutor()
        bar = BarEvent(
            symbol="EURUSD",
            timestamp=datetime(2024, 1, 1),
            open=1.10,
            high=1.16,
            low=1.099,
            close=1.15,
            volume=1.0,
        )
        result = exe.check_intrabar_stops(
            bar,
            OrderSide.BUY,
            stop_loss=1.095,
            take_profit=1.15,
        )
        self.assertEqual(result, ("target", 1.15))

    def test_both_hit_pessimistic_returns_stop(self):
        exe = SimulatedExecutor()
        # Wide bar that touches BOTH stop and target. We assume stop first.
        bar = BarEvent(
            symbol="EURUSD",
            timestamp=datetime(2024, 1, 1),
            open=1.10,
            high=1.16,
            low=1.09,
            close=1.10,
            volume=1.0,
        )
        result = exe.check_intrabar_stops(
            bar,
            OrderSide.BUY,
            stop_loss=1.095,
            take_profit=1.15,
        )
        self.assertEqual(result[0], "stop")

    def test_no_hit_returns_none(self):
        exe = SimulatedExecutor()
        bar = BarEvent(
            symbol="EURUSD",
            timestamp=datetime(2024, 1, 1),
            open=1.10,
            high=1.105,
            low=1.099,
            close=1.102,
            volume=1.0,
        )
        self.assertIsNone(
            exe.check_intrabar_stops(
                bar,
                OrderSide.BUY,
                stop_loss=1.095,
                take_profit=1.15,
            )
        )

    def test_short_position_stops_and_targets_inverted(self):
        exe = SimulatedExecutor()
        # Short: stop ABOVE entry, target BELOW. Bar high hits stop.
        bar = BarEvent(
            symbol="EURUSD",
            timestamp=datetime(2024, 1, 1),
            open=1.10,
            high=1.12,
            low=1.10,
            close=1.11,
            volume=1.0,
        )
        result = exe.check_intrabar_stops(
            bar,
            OrderSide.SELL,
            stop_loss=1.11,
            take_profit=1.08,
        )
        self.assertEqual(result, ("stop", 1.11))


if __name__ == "__main__":
    unittest.main()
