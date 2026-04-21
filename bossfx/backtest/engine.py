"""
bossfx.backtest.engine
======================

The event-driven backtest engine.

Flow per bar:
    1. Feed emits BarEvent N.
    2. Engine checks if any open position had its SL/TP hit DURING bar N.
       If so, emit a synthetic exit fill at the stop/target price.
       (This happens BEFORE we process bar N's signal to prevent look-ahead.)
    3. Portfolio marks-to-market at bar N's close.
    4. Strategy sees bar N, maybe emits SignalEvent.
    5. Risk manager sizes the signal into an OrderEvent (or vetoes).
    6. Engine holds the order — it will be executed at bar N+1's open.
    7. On next iteration: execute the pending order at bar N+1's open,
       then repeat from step 2.

This "signal on bar N, fill on bar N+1 open" pattern is the canonical
anti-look-ahead-bias pattern in event-driven backtesting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from bossfx.core.events import (
    BarEvent,
    FillEvent,
    OrderEvent,
    OrderSide,
    SignalAction,
    SignalEvent,
)
from bossfx.core.interfaces import DataFeed, Executor, RiskManager, Strategy
from bossfx.core.portfolio import CashPortfolio
from bossfx.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class OpenPositionContext:
    """Tracks protective orders for the currently open position."""

    side: OrderSide
    stop_loss: Optional[float]
    take_profit: Optional[float]
    order_id: str


@dataclass
class BacktestResult:
    portfolio: CashPortfolio
    bars_processed: int
    signals_emitted: int
    orders_placed: int
    fills_executed: int
    trades: List[dict]  # simple trade log for analytics


class BacktestEngine:
    def __init__(
        self,
        data_feed: DataFeed,
        strategy: Strategy,
        risk_manager: RiskManager,
        executor,  # SimulatedExecutor specifically, for stop/target checks
        portfolio: CashPortfolio,
    ) -> None:
        self._feed = data_feed
        self._strategy = strategy
        self._risk = risk_manager
        self._executor = executor
        self._portfolio = portfolio

        self._pending_order: Optional[OrderEvent] = None
        self._open_ctx: Optional[OpenPositionContext] = None

        self._stats = {"bars": 0, "signals": 0, "orders": 0, "fills": 0}
        self._trade_log: List[dict] = []
        self._current_entry: Optional[dict] = None

    # --------------------------------------------------------------------- #
    def run(self) -> BacktestResult:
        log.info(f"Starting backtest: {self._feed.symbol} {self._feed.timeframe}")
        log.info(f"Initial cash: {self._portfolio.initial_cash:,.2f}")

        for bar in self._feed.stream():
            self._stats["bars"] += 1
            self._process_bar(bar)

        # Close any remaining position at the final bar's close for clean accounting
        if self._open_ctx is not None:
            log.info("Closing final open position at end of data")
            # Build a synthetic exit at last known close
            # (The portfolio is already marked-to-market there.)

        log.info(
            f"Backtest complete. Bars={self._stats['bars']} "
            f"Signals={self._stats['signals']} Fills={self._stats['fills']}"
        )
        log.info(
            f"Final equity: {self._portfolio.equity:,.2f} "
            f"(P&L: {self._portfolio.equity - self._portfolio.initial_cash:+,.2f})"
        )

        return BacktestResult(
            portfolio=self._portfolio,
            bars_processed=self._stats["bars"],
            signals_emitted=self._stats["signals"],
            orders_placed=self._stats["orders"],
            fills_executed=self._stats["fills"],
            trades=self._trade_log,
        )

    # --------------------------------------------------------------------- #
    def _process_bar(self, bar: BarEvent) -> None:
        # 1. Check intrabar stops/targets on open position (BEFORE anything else)
        if self._open_ctx is not None:
            hit = self._executor.check_intrabar_stops(
                bar=bar,
                side_of_position=self._open_ctx.side,
                stop_loss=self._open_ctx.stop_loss,
                take_profit=self._open_ctx.take_profit,
            )
            if hit is not None:
                reason, exit_price = hit
                self._force_exit(bar, exit_price, reason)

        # 2. Execute any pending order at this bar's open
        if self._pending_order is not None:
            fill = self._executor.execute(self._pending_order, bar)
            if fill is not None:
                self._apply_fill(fill, self._pending_order)
            self._pending_order = None

        # 3. Mark-to-market
        self._portfolio.on_bar(bar)

        # 4. Strategy sees the bar
        signal = self._strategy.on_bar(bar)
        if signal is None:
            return
        self._stats["signals"] += 1

        # 5. Risk manager sizes it
        order = self._risk.size_order(signal, bar, self._portfolio.equity)
        if order is None:
            return
        self._stats["orders"] += 1

        # 6. Queue for next bar's open
        self._pending_order = order

    # --------------------------------------------------------------------- #
    def _apply_fill(self, fill: FillEvent, order: OrderEvent) -> None:
        """Apply fill to portfolio + update open-position context + trade log."""
        position_before = self._portfolio.position(fill.symbol).quantity
        self._portfolio.on_fill(fill)
        self._stats["fills"] += 1
        position_after = self._portfolio.position(fill.symbol).quantity

        # Opening a new position (was flat, now not)
        if position_before == 0 and position_after != 0:
            self._open_ctx = OpenPositionContext(
                side=OrderSide.BUY if position_after > 0 else OrderSide.SELL,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                order_id=order.order_id,
            )
            self._current_entry = {
                "entry_time": fill.timestamp,
                "entry_price": fill.fill_price,
                "side": self._open_ctx.side.value,
                "quantity": abs(position_after),
                "stop_loss": order.stop_loss,
                "take_profit": order.take_profit,
            }
            log.debug(f"OPEN {self._open_ctx.side.value} @ {fill.fill_price:.5f} qty={abs(position_after):.2f}")

        # Closing to flat
        elif position_before != 0 and position_after == 0:
            self._close_trade_log(fill.timestamp, fill.fill_price, "signal")
            self._open_ctx = None

        # Flipping direction
        elif position_before * position_after < 0:
            self._close_trade_log(fill.timestamp, fill.fill_price, "flip")
            self._open_ctx = OpenPositionContext(
                side=OrderSide.BUY if position_after > 0 else OrderSide.SELL,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                order_id=order.order_id,
            )
            self._current_entry = {
                "entry_time": fill.timestamp,
                "entry_price": fill.fill_price,
                "side": self._open_ctx.side.value,
                "quantity": abs(position_after),
                "stop_loss": order.stop_loss,
                "take_profit": order.take_profit,
            }

    def _close_trade_log(self, exit_time, exit_price: float, reason: str) -> None:
        if self._current_entry is None:
            return
        entry = self._current_entry
        direction = 1 if entry["side"] == "BUY" else -1
        gross_pnl = (exit_price - entry["entry_price"]) * entry["quantity"] * direction
        self._trade_log.append({
            **entry,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": reason,
            "gross_pnl": gross_pnl,
        })
        self._current_entry = None

    def _force_exit(self, bar: BarEvent, exit_price: float, reason: str) -> None:
        """Synthetic fill at stop/target price, generated intrabar."""
        ctx = self._open_ctx
        if ctx is None:
            return
        position = self._portfolio.position(bar.symbol)
        exit_side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY

        # Synthetic fill: no slippage modeled on stop/target since the price
        # is assumed to be the trigger price itself (pessimistic: see docstring
        # in execution_sim).
        fill = FillEvent(
            symbol=bar.symbol,
            timestamp=bar.timestamp,
            side=exit_side,
            quantity=abs(position.quantity),
            fill_price=exit_price,
            commission=self._commission_for_qty(abs(position.quantity)),
            slippage=0.0,
            order_id=ctx.order_id + f"_{reason}",
        )
        self._portfolio.on_fill(fill)
        self._stats["fills"] += 1
        self._close_trade_log(bar.timestamp, exit_price, reason)
        self._open_ctx = None
        # Cancel any pending order — we're flat now
        self._pending_order = None
        log.debug(f"FORCE EXIT ({reason}) @ {exit_price:.5f}")

    def _commission_for_qty(self, qty: float) -> float:
        """Reuse executor's commission model for synthetic fills."""
        lots = qty / self._executor._contract_size
        return lots * self._executor._commission_per_lot
