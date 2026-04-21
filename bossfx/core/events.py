"""
bossfx.core.events
==================

The four immutable event types that flow through BossFx.

Design philosophy
-----------------
Events are the *only* way components communicate. A strategy never calls
the portfolio directly; it emits a ``SignalEvent``. The portfolio never
calls the executor directly; it emits an ``OrderEvent``. This decoupling
is what makes the system swappable (backtest -> live) without rewrites.

All events are frozen dataclasses: once created, they cannot be mutated.
This is deliberate — it prevents subtle bugs where one component silently
changes data another component is holding a reference to.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    """Canonical event type tags (used for routing in the engine)."""

    BAR = "BAR"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class SignalAction(str, Enum):
    """What a strategy is telling the portfolio to do."""

    LONG = "LONG"           # open long position
    SHORT = "SHORT"         # open short position
    EXIT = "EXIT"           # close any open position
    HOLD = "HOLD"           # do nothing (emitted for audit trails)


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


# --------------------------------------------------------------------------- #
# BarEvent — a completed OHLCV candle                                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BarEvent:
    """A single completed price bar. Emitted by the data feed."""

    type: EventType = EventType.BAR
    symbol: str = ""
    timestamp: datetime = None            # type: ignore[assignment]
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    timeframe: str = "1H"

    def __post_init__(self) -> None:
        # Sanity checks — catch bad data at the source, not three layers deep.
        if self.high < self.low:
            raise ValueError(f"Bar {self.timestamp}: high < low ({self.high} < {self.low})")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"Bar {self.timestamp}: open {self.open} outside [{self.low}, {self.high}]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"Bar {self.timestamp}: close {self.close} outside [{self.low}, {self.high}]")


# --------------------------------------------------------------------------- #
# SignalEvent — strategy's intent                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SignalEvent:
    """A strategy's desire to enter/exit. Knows nothing about size or money."""

    type: EventType = EventType.SIGNAL
    symbol: str = ""
    timestamp: datetime = None            # type: ignore[assignment]
    action: SignalAction = SignalAction.HOLD
    strength: float = 1.0                 # 0.0–1.0, for future weighted ensembles
    strategy_id: str = ""                 # which strategy produced this
    reference_price: float = 0.0          # close price at signal time (for logging)
    metadata: Optional[dict] = None       # free-form context (e.g. indicator values)


# --------------------------------------------------------------------------- #
# OrderEvent — portfolio's instruction to the executor                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OrderEvent:
    """A concrete instruction: buy/sell X units of Y at Z conditions."""

    type: EventType = EventType.ORDER
    symbol: str = ""
    timestamp: datetime = None            # type: ignore[assignment]
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0                 # in units of base currency / lots
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    order_id: str = ""                    # UUID for reconciliation


# --------------------------------------------------------------------------- #
# FillEvent — execution confirmation                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FillEvent:
    """
    Confirmation from the executor that an order was filled (or partially filled).

    This is the *only* event that moves money in the portfolio. Nothing else
    should ever touch cash/equity. This makes accounting auditable.
    """

    type: EventType = EventType.FILL
    symbol: str = ""
    timestamp: datetime = None            # type: ignore[assignment]
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    fill_price: float = 0.0               # price after slippage
    commission: float = 0.0               # broker fee for this fill
    slippage: float = 0.0                 # difference from expected price
    order_id: str = ""
