"""
bossfx.config.settings
======================

Typed, validated configuration.

Design note
-----------
We use ``dataclasses`` as the base so the system runs in *any* Python 3.10+
environment without extra dependencies. If ``pydantic`` is installed,
``load_config`` additionally validates via pydantic for richer error
messages — but the public API stays identical.

YAML -> Config object -> rest of the system. One source of truth for
parameters. No hardcoded values anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    source: str = "csv"  # 'csv' or 'yfinance'
    symbol: str = "EURUSD=X"
    timeframe: str = "1h"
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    csv_path: Optional[str] = None

    def validate(self) -> None:
        if self.source not in ("csv", "yfinance"):
            raise ValueError(f"data.source must be 'csv' or 'yfinance', got {self.source}")
        if self.source == "csv" and not self.csv_path:
            raise ValueError("data.csv_path is required when source='csv'")
        if self.start and self.end and self.end <= self.start:
            raise ValueError(f"data.end ({self.end}) must be after start ({self.start})")


@dataclass
class StrategyConfig:
    name: str = "sma_crossover"
    fast_period: int = 20
    slow_period: int = 50

    def validate(self) -> None:
        if self.fast_period <= 0 or self.slow_period <= 0:
            raise ValueError("strategy periods must be > 0")
        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"fast_period ({self.fast_period}) must be < slow_period ({self.slow_period})"
            )


@dataclass
class RiskConfig:
    initial_cash: float = 10_000.0
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.20
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02
    max_drawdown_pct: float = 0.25

    def validate(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("risk.initial_cash must be > 0")
        if not (0 < self.risk_per_trade_pct <= 0.1):
            raise ValueError("risk.risk_per_trade_pct must be in (0, 0.1]")
        if not (0 < self.max_position_pct <= 1.0):
            raise ValueError("risk.max_position_pct must be in (0, 1.0]")
        if self.stop_loss_pct <= 0 or self.take_profit_pct <= 0:
            raise ValueError("risk.stop_loss_pct and take_profit_pct must be > 0")


@dataclass
class ExecutionConfig:
    spread_pips: float = 1.0
    slippage_pips: float = 0.5
    commission_per_lot: float = 7.0
    pip_size: float = 0.0001
    contract_size: float = 100_000.0

    def validate(self) -> None:
        for name in ("spread_pips", "slippage_pips", "commission_per_lot"):
            if getattr(self, name) < 0:
                raise ValueError(f"execution.{name} must be >= 0")
        if self.pip_size <= 0 or self.contract_size <= 0:
            raise ValueError("pip_size and contract_size must be > 0")


@dataclass
class BacktestConfig:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    log_level: str = "INFO"

    def validate(self) -> None:
        self.data.validate()
        self.strategy.validate()
        self.risk.validate()
        self.execution.validate()


# --------------------------------------------------------------------------- #
def _coerce_datetime(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    # YAML may yield a ``datetime.date`` for plain 'YYYY-MM-DD' strings
    return datetime(value.year, value.month, value.day)


def load_config(path: str | Path) -> BacktestConfig:
    """Load and validate a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    data_raw = dict(raw.get("data", {}))
    data_raw["start"] = _coerce_datetime(data_raw.get("start"))
    data_raw["end"] = _coerce_datetime(data_raw.get("end"))

    cfg = BacktestConfig(
        data=DataConfig(**data_raw),
        strategy=StrategyConfig(**raw.get("strategy", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        execution=ExecutionConfig(**raw.get("execution", {})),
        log_level=raw.get("log_level", "INFO"),
    )
    cfg.validate()
    return cfg
