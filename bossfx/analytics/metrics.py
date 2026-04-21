"""
bossfx.analytics.metrics
========================

The metrics that actually matter.

A strategy with +20% return and 60% max drawdown is WORSE than one with
+12% return and 8% drawdown. Total return alone is a terrible way to
evaluate a strategy. Here's what matters instead:

* **Sharpe Ratio**: return per unit of volatility (risk-adjusted return).
* **Sortino Ratio**: like Sharpe, but only penalizes *downside* volatility.
* **Max Drawdown**: the worst peak-to-trough loss. This is what kills accounts.
* **Calmar Ratio**: annualized return / max drawdown. "Return per unit of pain."
* **Win Rate**: % of trades that were profitable.
* **Profit Factor**: gross wins / gross losses. >1.5 is good, >2 is great.
* **Expectancy**: average $ gained per trade. The long-run truth-teller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

# Standard annualization factors for common timeframes
ANNUALIZATION = {
    "1m": 252 * 24 * 60,
    "5m": 252 * 24 * 12,
    "15m": 252 * 24 * 4,
    "1h": 252 * 24,
    "4h": 252 * 6,
    "1d": 252,
}


@dataclass
class PerformanceReport:
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    volatility_pct: float
    num_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    def pretty(self) -> str:
        lines = [
            "╔════════════════════════════════════════════════╗",
            "║          BossFx Performance Report             ║",
            "╠════════════════════════════════════════════════╣",
            f"║ Total Return         : {self.total_return_pct:+10.2f}%         ║",
            f"║ CAGR                 : {self.cagr_pct:+10.2f}%         ║",
            f"║ Volatility (annual)  : {self.volatility_pct:10.2f}%         ║",
            f"║ Sharpe Ratio         : {self.sharpe:10.2f}          ║",
            f"║ Sortino Ratio        : {self.sortino:10.2f}          ║",
            f"║ Calmar Ratio         : {self.calmar:10.2f}          ║",
            f"║ Max Drawdown         : {self.max_drawdown_pct:10.2f}%         ║",
            f"║ Max DD Duration      : {self.max_drawdown_duration_bars:10d} bars    ║",
            "╠════════════════════════════════════════════════╣",
            f"║ Total Trades         : {self.num_trades:10d}          ║",
            f"║ Win Rate             : {self.win_rate_pct:10.2f}%         ║",
            f"║ Profit Factor        : {self.profit_factor:10.2f}          ║",
            f"║ Expectancy / Trade   : {self.expectancy:+10.2f}          ║",
            f"║ Avg Win              : {self.avg_win:+10.2f}          ║",
            f"║ Avg Loss             : {self.avg_loss:+10.2f}          ║",
            f"║ Largest Win          : {self.largest_win:+10.2f}          ║",
            f"║ Largest Loss         : {self.largest_loss:+10.2f}          ║",
            "╚════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def compute_report(
    equity_curve: List[Tuple],
    trades: List[dict],
    initial_cash: float,
    timeframe: str = "1h",
) -> PerformanceReport:
    if not equity_curve:
        raise ValueError("Empty equity curve")

    df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    equity = df["equity"].astype(float)
    returns = equity.pct_change().fillna(0.0)

    # ---- Returns ----
    total_return_pct = (equity.iloc[-1] / initial_cash - 1) * 100

    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    years = max(span_days / 365.25, 1e-9)
    cagr_pct = ((equity.iloc[-1] / initial_cash) ** (1 / years) - 1) * 100 if equity.iloc[-1] > 0 else -100.0

    # ---- Volatility (annualized) ----
    ann_factor = ANNUALIZATION.get(timeframe.lower(), 252)
    vol_annual = returns.std() * math.sqrt(ann_factor) if len(returns) > 1 else 0.0
    volatility_pct = vol_annual * 100

    # ---- Sharpe (assume 0 risk-free for simplicity; swap in Phase 3) ----
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * math.sqrt(ann_factor) if std_ret > 0 else 0.0

    # ---- Sortino (downside deviation only) ----
    downside = returns[returns < 0]
    dd_std = downside.std()
    sortino = (mean_ret / dd_std) * math.sqrt(ann_factor) if dd_std and dd_std > 0 else 0.0

    # ---- Max drawdown ----
    peak = equity.cummax()
    dd_series = (equity - peak) / peak
    max_dd = dd_series.min()
    max_drawdown_pct = max_dd * 100

    # Duration of worst drawdown (in bars)
    in_dd = dd_series < 0
    max_dd_duration = 0
    current = 0
    for flag in in_dd:
        if flag:
            current += 1
            max_dd_duration = max(max_dd_duration, current)
        else:
            current = 0

    # ---- Calmar ----
    calmar = (cagr_pct / 100) / abs(max_dd) if max_dd < 0 else 0.0

    # ---- Trade stats ----
    num_trades = len(trades)
    if num_trades > 0:
        pnls = np.array([t["gross_pnl"] for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate_pct = (len(wins) / num_trades) * 100
        profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")
        expectancy = pnls.mean()
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0
        largest_win = wins.max() if len(wins) > 0 else 0.0
        largest_loss = losses.min() if len(losses) > 0 else 0.0
    else:
        win_rate_pct = profit_factor = expectancy = 0.0
        avg_win = avg_loss = largest_win = largest_loss = 0.0

    return PerformanceReport(
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_duration_bars=max_dd_duration,
        volatility_pct=volatility_pct,
        num_trades=num_trades,
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor if not math.isinf(profit_factor) else 999.99,
        expectancy=expectancy,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
    )
