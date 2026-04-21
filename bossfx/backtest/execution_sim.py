"""
bossfx.backtest.execution_sim
=============================

Realistic execution simulator.

Every backtest that says "assume zero spread, zero slippage, zero commission"
is lying to you. Forex has spreads. Orders don't always fill at the price
you clicked. Brokers charge fees. This simulator models all three.

Philosophy: err *pessimistic*. If your strategy only makes money after we
apply a pessimistic execution model, then small improvements in reality
(tighter spread, better broker) are upside — not required-for-survival.
"""
from __future__ import annotations

from typing import Optional

from bossfx.core.events import BarEvent, FillEvent, OrderEvent, OrderSide, OrderType
from bossfx.core.interfaces import Executor
from bossfx.utils.logger import get_logger

log = get_logger(__name__)


class SimulatedExecutor(Executor):
    """
    Simulates fills on the *next* bar's open — this is the correct anti-
    look-ahead pattern. When a strategy emits a signal on bar N's close,
    the fill happens at bar N+1's open, because that's the earliest moment
    a real order could reach the broker.
    """

    def __init__(
        self,
        spread_pips: float = 1.0,
        slippage_pips: float = 0.5,
        commission_per_lot: float = 7.0,
        pip_size: float = 0.0001,
        contract_size: float = 100_000.0,
    ) -> None:
        self._spread = spread_pips * pip_size
        self._slippage = slippage_pips * pip_size
        self._commission_per_lot = commission_per_lot
        self._pip_size = pip_size
        self._contract_size = contract_size

    def execute(self, order: OrderEvent, bar: BarEvent) -> Optional[FillEvent]:
        """
        ``bar`` here is the bar AFTER the signal was generated. The caller
        (the engine) is responsible for this sequencing — we just fill at
        this bar's open with realistic costs baked in.
        """
        if order.order_type != OrderType.MARKET:
            log.warning(f"Non-market orders not yet supported (got {order.order_type})")
            return None

        # Base fill price is the open of the next bar
        base_price = bar.open
        half_spread = self._spread / 2

        # Buyers pay the ask (above mid), sellers receive the bid (below mid)
        if order.side == OrderSide.BUY:
            fill_price = base_price + half_spread + self._slippage
        else:
            fill_price = base_price - half_spread - self._slippage

        # Commission proportional to lot size (quantity / contract_size)
        lots = order.quantity / self._contract_size
        commission = abs(lots) * self._commission_per_lot

        slippage_cost = abs(fill_price - base_price)

        return FillEvent(
            symbol=order.symbol,
            timestamp=bar.timestamp,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_cost,
            order_id=order.order_id,
        )

    def check_intrabar_stops(
        self,
        bar: BarEvent,
        side_of_position: OrderSide,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> Optional[tuple[str, float]]:
        """
        Check if a bar's high/low would have triggered our SL or TP.

        Returns ('stop', price) or ('target', price) or None.

        This is the honest way to handle intrabar exits in a bar-based
        backtest. We assume the worst case when both could have hit in
        the same bar: the stop triggers first (again — pessimism).
        """
        if stop_loss is None and take_profit is None:
            return None

        hit_stop = False
        hit_target = False

        if side_of_position == OrderSide.BUY:  # long position
            if stop_loss is not None and bar.low <= stop_loss:
                hit_stop = True
            if take_profit is not None and bar.high >= take_profit:
                hit_target = True
        else:  # short position
            if stop_loss is not None and bar.high >= stop_loss:
                hit_stop = True
            if take_profit is not None and bar.low <= take_profit:
                hit_target = True

        if hit_stop and hit_target:
            # Pessimistic: assume stop hit first
            return ("stop", stop_loss)  # type: ignore[return-value]
        if hit_stop:
            return ("stop", stop_loss)  # type: ignore[return-value]
        if hit_target:
            return ("target", take_profit)  # type: ignore[return-value]
        return None
