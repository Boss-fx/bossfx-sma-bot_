"""
Tests for CashPortfolio — TIER 1 (accounting invariants).

The sacred invariant:
    equity  ==  cash + sum(unrealized P&L across positions)

If this is ever wrong by a cent, the whole system lies. Money
disappearing in the accounting layer is the worst possible bug in a
trading system — worse than a crash, because it fails silently.

Every test here exists to defend against that.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from bossfx.core.events import BarEvent, FillEvent, OrderSide
from bossfx.core.portfolio import CashPortfolio


def _bar(sym, ts, close, high=None, low=None):
    return BarEvent(
        symbol=sym,
        timestamp=ts,
        open=close,
        high=high or close + 0.001,
        low=low or close - 0.001,
        close=close,
        volume=100.0,
    )


def _fill(sym, ts, side, qty, price, commission=0.0):
    return FillEvent(
        symbol=sym,
        timestamp=ts,
        side=side,
        quantity=qty,
        fill_price=price,
        commission=commission,
        slippage=0.0,
        order_id="t",
    )


class TestInitialization(unittest.TestCase):
    def test_initial_cash_equals_equity(self):
        p = CashPortfolio(initial_cash=10_000)
        self.assertEqual(p.cash, 10_000)
        self.assertEqual(p.equity, 10_000)

    def test_rejects_non_positive_cash(self):
        with self.assertRaises(ValueError):
            CashPortfolio(initial_cash=0)
        with self.assertRaises(ValueError):
            CashPortfolio(initial_cash=-1)


class TestLongRoundTrip(unittest.TestCase):
    """Buy 1000 @ 1.10, mark-to-market, sell 1000 @ 1.12 -> +20 P&L."""

    def setUp(self):
        self.p = CashPortfolio(initial_cash=10_000)
        self.t0 = datetime(2024, 1, 1, 0, 0)

    def test_open_long_does_not_change_cash_other_than_commission(self):
        # Opening a long doesn't pay out cash in this simplified model
        # (we track P&L only on close). Commission IS taken out.
        self.p.on_fill(_fill("EURUSD", self.t0, OrderSide.BUY, 1000, 1.10, commission=2.0))
        self.assertAlmostEqual(self.p.cash, 10_000 - 2.0, places=6)
        pos = self.p.position("EURUSD")
        self.assertEqual(pos.quantity, 1000)
        self.assertAlmostEqual(pos.avg_price, 1.10, places=6)

    def test_mark_to_market_updates_equity_not_cash(self):
        self.p.on_fill(_fill("EURUSD", self.t0, OrderSide.BUY, 1000, 1.10))
        self.p.on_bar(_bar("EURUSD", self.t0 + timedelta(hours=1), 1.12))
        # Unrealized P&L = (1.12 - 1.10) * 1000 = 20
        self.assertAlmostEqual(self.p.equity, 10_020, places=4)
        self.assertAlmostEqual(self.p.cash, 10_000, places=4)

    def test_close_realizes_pnl(self):
        self.p.on_fill(_fill("EURUSD", self.t0, OrderSide.BUY, 1000, 1.10))
        self.p.on_bar(_bar("EURUSD", self.t0 + timedelta(hours=1), 1.12))
        self.p.on_fill(_fill("EURUSD", self.t0 + timedelta(hours=2), OrderSide.SELL, 1000, 1.12))
        # After close, cash = 10_000 + 20 = 10_020. Position is flat.
        self.assertAlmostEqual(self.p.cash, 10_020, places=4)
        self.assertTrue(self.p.position("EURUSD").is_flat())


class TestShortRoundTrip(unittest.TestCase):
    """Sell 1000 @ 1.12, buy back 1000 @ 1.10 -> +20 P&L."""

    def test_short_profits_when_price_falls(self):
        p = CashPortfolio(initial_cash=10_000)
        t0 = datetime(2024, 1, 1)
        p.on_fill(_fill("EURUSD", t0, OrderSide.SELL, 1000, 1.12))
        p.on_bar(_bar("EURUSD", t0 + timedelta(hours=1), 1.10))
        self.assertAlmostEqual(p.equity, 10_020, places=4)
        p.on_fill(_fill("EURUSD", t0 + timedelta(hours=2), OrderSide.BUY, 1000, 1.10))
        self.assertAlmostEqual(p.cash, 10_020, places=4)
        self.assertTrue(p.position("EURUSD").is_flat())

    def test_short_loses_when_price_rises(self):
        p = CashPortfolio(initial_cash=10_000)
        t0 = datetime(2024, 1, 1)
        p.on_fill(_fill("EURUSD", t0, OrderSide.SELL, 1000, 1.10))
        p.on_fill(_fill("EURUSD", t0 + timedelta(hours=1), OrderSide.BUY, 1000, 1.12))
        # Lost 20
        self.assertAlmostEqual(p.cash, 9_980, places=4)


class TestPositionFlip(unittest.TestCase):
    """Holding long, a SELL larger than the long should close + open short."""

    def test_flip_long_to_short(self):
        p = CashPortfolio(initial_cash=10_000)
        t0 = datetime(2024, 1, 1)
        # Open long 500 @ 1.10
        p.on_fill(_fill("EURUSD", t0, OrderSide.BUY, 500, 1.10))
        # Sell 800 @ 1.12 -> closes 500 (+10 profit) and opens short 300 @ 1.12
        p.on_fill(_fill("EURUSD", t0 + timedelta(hours=1), OrderSide.SELL, 800, 1.12))
        pos = p.position("EURUSD")
        self.assertAlmostEqual(pos.quantity, -300, places=4)
        self.assertAlmostEqual(pos.avg_price, 1.12, places=4)
        self.assertAlmostEqual(p.cash, 10_010, places=4)  # realized +10


class TestAccountingInvariant(unittest.TestCase):
    """The invariant: equity == cash + unrealized P&L, ALWAYS."""

    def test_invariant_holds_across_many_fills(self):
        p = CashPortfolio(initial_cash=10_000)
        t0 = datetime(2024, 1, 1)
        # Build up a position in 3 adds
        prices = [1.10, 1.11, 1.12]
        for i, px in enumerate(prices):
            p.on_fill(
                _fill("EURUSD", t0 + timedelta(hours=i), OrderSide.BUY, 100, px, commission=0.5)
            )
        p.on_bar(_bar("EURUSD", t0 + timedelta(hours=3), 1.15))
        pos = p.position("EURUSD")
        expected_avg = (1.10 + 1.11 + 1.12) / 3
        self.assertAlmostEqual(pos.avg_price, expected_avg, places=4)
        self.assertAlmostEqual(pos.quantity, 300, places=4)

        expected_unrealized = (1.15 - expected_avg) * 300
        expected_cash = 10_000 - 0.5 * 3
        self.assertAlmostEqual(p.equity, expected_cash + expected_unrealized, places=4)

    def test_commissions_reduce_cash_even_on_losing_trade(self):
        p = CashPortfolio(initial_cash=10_000)
        t0 = datetime(2024, 1, 1)
        p.on_fill(_fill("EURUSD", t0, OrderSide.BUY, 1000, 1.10, commission=3.0))
        p.on_fill(
            _fill("EURUSD", t0 + timedelta(hours=1), OrderSide.SELL, 1000, 1.09, commission=3.0)
        )
        # Lost 10 on price + 6 commission = 9_984
        self.assertAlmostEqual(p.cash, 9_984, places=4)


class TestEquityCurve(unittest.TestCase):
    def test_equity_curve_records_every_bar(self):
        p = CashPortfolio(initial_cash=10_000)
        t0 = datetime(2024, 1, 1)
        for i in range(5):
            p.on_bar(_bar("EURUSD", t0 + timedelta(hours=i), 1.10))
        self.assertEqual(len(p.equity_curve()), 5)


if __name__ == "__main__":
    unittest.main()
