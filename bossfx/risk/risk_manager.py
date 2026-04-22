"""
bossfx.risk.risk_manager
========================

Percent-risk position sizing.

The math
--------
If your stop is $0.50 away and you risk $100 per trade, you buy 200 units.
In general:
    quantity = risk_cash / stop_distance_per_unit

where:
    risk_cash = equity * risk_per_trade_pct
    stop_distance_per_unit = abs(entry_price - stop_price)

This is the single most important change vs. the original bot. Fixed-size
trading compounds badly in drawdowns (you lose 50%, you now need +100% to
recover). Percent-risk sizing means losses shrink your position naturally.

Phase 1 uses simple price-% stops. Phase 2 will plug in ATR-based stops.
"""

from __future__ import annotations

import uuid
from typing import Optional

from bossfx.core.events import (
    BarEvent,
    OrderEvent,
    OrderSide,
    OrderType,
    SignalAction,
    SignalEvent,
)
from bossfx.core.interfaces import RiskManager
from bossfx.core.portfolio import CashPortfolio
from bossfx.utils.logger import get_logger

log = get_logger(__name__)


class PercentRiskManager(RiskManager):
    def __init__(
        self,
        portfolio: CashPortfolio,
        risk_per_trade_pct: float = 0.01,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float = 0.02,
        max_position_pct: float = 0.20,
        max_drawdown_pct: float = 0.25,
    ) -> None:
        self._portfolio = portfolio
        self._risk_per_trade = risk_per_trade_pct
        self._sl_pct = stop_loss_pct
        self._tp_pct = take_profit_pct
        self._max_position_pct = max_position_pct
        self._max_drawdown_pct = max_drawdown_pct
        self._peak_equity = portfolio.equity
        self._halted = False

    # --------------------------------------------------------------------- #
    def size_order(
        self,
        signal: SignalEvent,
        bar: BarEvent,
        equity: float,
    ) -> Optional[OrderEvent]:
        """Return a sized OrderEvent, or None if we veto the trade."""
        # ---- Circuit breaker: drawdown cap ------------------------------ #
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (self._peak_equity - equity) / self._peak_equity
        if drawdown >= self._max_drawdown_pct:
            if not self._halted:
                log.warning(
                    f"RISK HALT: drawdown {drawdown:.1%} exceeded cap "
                    f"{self._max_drawdown_pct:.1%}. No new trades."
                )
                self._halted = True
            return None

        # ---- Handle EXIT signals directly ------------------------------- #
        position = self._portfolio.position(signal.symbol)
        if signal.action == SignalAction.EXIT:
            if position.is_flat():
                return None
            side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
            return OrderEvent(
                symbol=signal.symbol,
                timestamp=signal.timestamp,
                side=side,
                order_type=OrderType.MARKET,
                quantity=abs(position.quantity),
                order_id=str(uuid.uuid4()),
            )

        # ---- Handle LONG / SHORT signals -------------------------------- #
        if signal.action not in (SignalAction.LONG, SignalAction.SHORT):
            return None

        entry_price = bar.close  # will be adjusted for spread/slippage by executor
        if entry_price <= 0:
            return None

        is_long = signal.action == SignalAction.LONG
        side = OrderSide.BUY if is_long else OrderSide.SELL

        # Stop & target prices
        if is_long:
            stop_price = entry_price * (1 - self._sl_pct)
            tp_price = entry_price * (1 + self._tp_pct)
        else:
            stop_price = entry_price * (1 + self._sl_pct)
            tp_price = entry_price * (1 - self._tp_pct)

        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return None

        # The core position-sizing equation
        risk_cash = equity * self._risk_per_trade
        quantity = risk_cash / stop_distance

        # Cap notional exposure so one bad trade can't blow us up even if
        # our stop calculation is wrong. 20% of equity max per position.
        max_notional = equity * self._max_position_pct
        if quantity * entry_price > max_notional:
            quantity = max_notional / entry_price

        # If we already have a position in the opposite direction, we need
        # to close it first. For Phase 1 we flip in a single order (the
        # portfolio handles reversal math). Phase 2 will split into two.
        if not position.is_flat():
            if (is_long and position.quantity > 0) or (not is_long and position.quantity < 0):
                # Same direction — ignore (don't pyramid in Phase 1)
                log.debug(f"Ignoring {signal.action} — already in same-side position")
                return None
            # Opposite direction — flip: close + open in one order
            quantity += abs(position.quantity)

        if quantity <= 0:
            return None

        return OrderEvent(
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            stop_loss=stop_price,
            take_profit=tp_price,
            order_id=str(uuid.uuid4()),
        )
