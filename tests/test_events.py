"""Tests for core event objects — immutability & validation."""
from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime

from bossfx.core.events import (
    BarEvent, EventType, FillEvent, OrderEvent, OrderSide,
    OrderType, SignalAction, SignalEvent,
)


class TestBarEvent(unittest.TestCase):
    def test_valid_bar_constructs(self):
        bar = BarEvent(
            symbol="EURUSD", timestamp=datetime(2024, 1, 1),
            open=1.10, high=1.12, low=1.09, close=1.11, volume=1000.0,
        )
        self.assertEqual(bar.symbol, "EURUSD")
        self.assertEqual(bar.type, EventType.BAR)

    def test_bar_is_immutable(self):
        bar = BarEvent(symbol="EURUSD", timestamp=datetime(2024, 1, 1),
                       open=1.10, high=1.12, low=1.09, close=1.11, volume=1000.0)
        with self.assertRaises(FrozenInstanceError):
            bar.close = 9.99   # type: ignore[misc]

    def test_rejects_high_below_low(self):
        with self.assertRaises(ValueError):
            BarEvent(symbol="EURUSD", timestamp=datetime(2024, 1, 1),
                     open=1.10, high=1.08, low=1.12, close=1.11, volume=1.0)

    def test_rejects_open_outside_range(self):
        with self.assertRaises(ValueError):
            BarEvent(symbol="EURUSD", timestamp=datetime(2024, 1, 1),
                     open=1.50, high=1.12, low=1.09, close=1.11, volume=1.0)

    def test_rejects_close_outside_range(self):
        with self.assertRaises(ValueError):
            BarEvent(symbol="EURUSD", timestamp=datetime(2024, 1, 1),
                     open=1.10, high=1.12, low=1.09, close=0.50, volume=1.0)


class TestSignalEvent(unittest.TestCase):
    def test_defaults_to_hold(self):
        s = SignalEvent(symbol="EURUSD", timestamp=datetime(2024, 1, 1))
        self.assertEqual(s.action, SignalAction.HOLD)

    def test_is_immutable(self):
        s = SignalEvent(symbol="EURUSD", timestamp=datetime(2024, 1, 1),
                        action=SignalAction.LONG)
        with self.assertRaises(FrozenInstanceError):
            s.action = SignalAction.SHORT   # type: ignore[misc]


class TestOrderAndFill(unittest.TestCase):
    def test_order_constructs(self):
        o = OrderEvent(symbol="EURUSD", timestamp=datetime(2024, 1, 1),
                       side=OrderSide.BUY, order_type=OrderType.MARKET,
                       quantity=1000, stop_loss=1.08, take_profit=1.14)
        self.assertEqual(o.side, OrderSide.BUY)

    def test_fill_constructs(self):
        f = FillEvent(symbol="EURUSD", timestamp=datetime(2024, 1, 1),
                      side=OrderSide.BUY, quantity=1000, fill_price=1.10,
                      commission=2.0, slippage=0.0001, order_id="abc")
        self.assertEqual(f.quantity, 1000)
        self.assertEqual(f.commission, 2.0)


if __name__ == "__main__":
    unittest.main()
