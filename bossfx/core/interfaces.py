"""
bossfx.core.interfaces
======================

Abstract base classes (ABCs) defining the contracts that every pluggable
component must honor.

Think of these as the *shape of the plug*. You can swap any USB device for
any other USB device because they all honor the USB shape. Similarly:

* Any ``DataFeed`` can be swapped for any other (yfinance -> MT5 -> CSV).
* Any ``Strategy`` can be swapped for any other.
* Any ``Executor`` can be swapped for any other (backtest sim -> live broker).

This is the *Dependency Inversion Principle* in practice — and it's what
will let you sell this as SaaS one day without rewriting the core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from bossfx.core.events import BarEvent, FillEvent, OrderEvent, SignalEvent


# --------------------------------------------------------------------------- #
class DataFeed(ABC):
    """A source of BarEvents. Must yield them in strict chronological order."""

    @abstractmethod
    def stream(self) -> Iterator[BarEvent]:
        """Yield BarEvents one at a time, oldest first. Infinite for live feeds."""
        ...

    @property
    @abstractmethod
    def symbol(self) -> str: ...

    @property
    @abstractmethod
    def timeframe(self) -> str: ...


# --------------------------------------------------------------------------- #
class Strategy(ABC):
    """Consumes BarEvents, produces SignalEvents. Stateless w.r.t. portfolio."""

    strategy_id: str = "base"

    @abstractmethod
    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        """
        Process a single bar. Return a SignalEvent or None.

        IMPORTANT: A strategy must never peek at future bars. It receives bars
        one at a time, in order. If you need to look back (e.g., for an SMA),
        buffer past bars internally.
        """
        ...

    @abstractmethod
    def warmup_period(self) -> int:
        """How many bars we need before signals are valid (e.g., slow SMA period)."""
        ...


# --------------------------------------------------------------------------- #
class RiskManager(ABC):
    """Vetoes or transforms signals into orders with appropriate sizing."""

    @abstractmethod
    def size_order(
        self,
        signal: SignalEvent,
        bar: BarEvent,
        equity: float,
    ) -> Optional[OrderEvent]:
        """Convert a signal to a sized order, or return None to veto."""
        ...


# --------------------------------------------------------------------------- #
class Executor(ABC):
    """Converts OrderEvents into FillEvents. Backtest sim or live broker."""

    @abstractmethod
    def execute(self, order: OrderEvent, bar: BarEvent) -> Optional[FillEvent]:
        """Attempt to execute ``order`` given current market context ``bar``."""
        ...


# --------------------------------------------------------------------------- #
class Portfolio(ABC):
    """Tracks positions, cash, equity. Single source of truth for account state."""

    @abstractmethod
    def on_fill(self, fill: FillEvent) -> None: ...

    @abstractmethod
    def on_bar(self, bar: BarEvent) -> None:
        """Mark-to-market: update unrealized P&L based on latest close."""
        ...

    @property
    @abstractmethod
    def equity(self) -> float:
        """Cash + unrealized P&L. The number we report to the user."""
        ...

    @property
    @abstractmethod
    def cash(self) -> float: ...

    @abstractmethod
    def equity_curve(self) -> List[tuple]:
        """List of (timestamp, equity) tuples — for analytics."""
        ...
