"""
bossfx.core.portfolio
=====================

The portfolio is the *single source of truth* for your account state.

Every fill goes through here. Every equity query comes from here. Every
P&L calculation happens here. Keeping this concentrated in one place means
you can audit every dollar that moves — and auditability is what separates
a toy from a fintech product.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

from bossfx.core.events import BarEvent, FillEvent, OrderSide
from bossfx.core.interfaces import Portfolio


@dataclass
class Position:
    """An open position in a single symbol."""

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.quantity == 0:
            return 0.0
        return (mark_price - self.avg_price) * self.quantity

    def is_flat(self) -> bool:
        return self.quantity == 0.0


class CashPortfolio(Portfolio):
    """
    A straightforward cash-account portfolio.

    Simplifying assumptions (to be relaxed in later phases):
      * No leverage / margin modeling yet.
      * One position per symbol (no hedging mode).
      * Quote currency == account currency.
    """

    def __init__(self, initial_cash: float = 10_000.0) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._positions: Dict[str, Position] = {}
        self._last_mark: Dict[str, float] = {}
        self._equity_curve: List[Tuple[datetime, float]] = []
        self._fills: List[FillEvent] = []

    def on_fill(self, fill: FillEvent) -> None:
        """Apply a fill to cash + position. The *only* way money moves."""
        self._fills.append(fill)
        pos = self._positions.setdefault(fill.symbol, Position(symbol=fill.symbol))

        signed_qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity

        # Case A: opening or adding in same direction
        if pos.quantity == 0 or (pos.quantity > 0 and signed_qty > 0) or (pos.quantity < 0 and signed_qty < 0):
            new_qty = pos.quantity + signed_qty
            pos.avg_price = (
                (pos.avg_price * abs(pos.quantity) + fill.fill_price * abs(signed_qty))
                / abs(new_qty)
            )
            pos.quantity = new_qty

        # Case B: reducing or flipping position
        else:
            closing_qty = min(abs(signed_qty), abs(pos.quantity))
            direction = 1 if pos.quantity > 0 else -1
            pnl = (fill.fill_price - pos.avg_price) * closing_qty * direction
            pos.realized_pnl += pnl
            self._cash += pnl

            remaining = signed_qty + (closing_qty * direction)
            if abs(remaining) < 1e-9:
                pos.quantity = 0.0
                pos.avg_price = 0.0
            else:
                pos.quantity = remaining
                pos.avg_price = fill.fill_price

        self._cash -= fill.commission

    def on_bar(self, bar: BarEvent) -> None:
        """Mark-to-market at bar close for reporting."""
        self._last_mark[bar.symbol] = bar.close
        self._equity_curve.append((bar.timestamp, self.equity))

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def equity(self) -> float:
        unreal = sum(
            pos.unrealized_pnl(self._last_mark.get(pos.symbol, pos.avg_price))
            for pos in self._positions.values()
        )
        return self._cash + unreal

    @property
    def initial_cash(self) -> float:
        return self._initial_cash

    def position(self, symbol: str) -> Position:
        return self._positions.setdefault(symbol, Position(symbol=symbol))

    def equity_curve(self) -> List[Tuple[datetime, float]]:
        return list(self._equity_curve)

    def fills(self) -> List[FillEvent]:
        return list(self._fills)
