# BossFx — Algorithmic Trading Framework

> A modular, event-driven algorithmic trading system designed for honest
> backtesting, production-ready execution, and SaaS-scale extensibility.

**Status:** Phase 1 complete — foundation refactor. 57 tests, all passing.
See the [Roadmap](#roadmap) for what's coming next.

---

## Why BossFx exists

Most retail trading bots lie to you. Not maliciously — structurally.
They use vectorized backtests that leak future data, ignore spread and
slippage, and assume you can size every trade the same regardless of
volatility or drawdown. Their backtests look great; their live accounts
blow up.

BossFx is built on a different principle: **the backtest must behave
exactly like live trading, or it's worthless.** That means:

- **Event-driven architecture** — every price bar is processed in
  strict chronological order. No component can ever see the future.
- **Online indicators** — moving averages, ATR, etc. are stateful
  objects that *cannot* be fed future data. Look-ahead bias becomes
  structurally impossible, not just unlikely.
- **Realistic execution modeling** — spread, slippage, and commissions
  are applied to every fill. If the strategy only works without them,
  we want to know.
- **Signal-on-bar-N, fill-on-bar-N+1 open** — the canonical
  anti-look-ahead pattern. Orders cannot be filled on the bar that
  produced them.

---

## Architecture at a glance

BossFx uses a hexagonal (ports-and-adapters) architecture. Every layer
has an abstract contract, so components are swappable without rewrites.

```
         ┌─────────────┐
         │  DataFeed   │   CSV, yfinance, MT5, Polygon, ...
         └──────┬──────┘
                │ BarEvent
                ▼
         ┌─────────────┐
         │  Strategy   │   SMA crossover today; multi-factor tomorrow
         └──────┬──────┘
                │ SignalEvent
                ▼
         ┌─────────────┐
         │ RiskManager │   Percent-risk sizing + drawdown circuit breaker
         └──────┬──────┘
                │ OrderEvent
                ▼
         ┌─────────────┐
         │  Executor   │   SimulatedExecutor (backtest) or MT5Executor (live)
         └──────┬──────┘
                │ FillEvent
                ▼
         ┌─────────────┐
         │  Portfolio  │   Single source of truth for cash, positions, equity
         └──────┬──────┘
                │ equity_curve + trade_log
                ▼
         ┌─────────────┐
         │  Analytics  │   Sharpe, Sortino, Calmar, drawdown, profit factor
         └─────────────┘
```

**Swap any block; the others don't care.** This is what makes the system
ready to become a SaaS platform: when you wire in a live MT5 executor,
the backtest engine doesn't change one line.

---

## Quickstart

### Install

```bash
git clone https://github.com/YOUR_ORG/bossfx.git
cd bossfx
pip install -r requirements.txt
# or for development:
pip install -e ".[dev]"
```

### Run a backtest

```bash
python -m scripts.run_backtest --config configs/eurusd_sma_default.yaml
```

Output:

```
╔════════════════════════════════════════════════╗
║          BossFx Performance Report             ║
╠════════════════════════════════════════════════╣
║ Total Return         :      +2.76%         ║
║ CAGR                 :     +12.70%         ║
║ Sharpe Ratio         :      15.77          ║
║ Sortino Ratio        :      20.35          ║
║ Max Drawdown         :      -0.23%         ║
║ Total Trades         :         37          ║
║ Win Rate             :      62.16%         ║
║ Profit Factor        :       3.32          ║
╚════════════════════════════════════════════════╝
```

### Run the tests

```bash
python -m unittest discover tests -v
```

57 tests covering accounting invariants, no-look-ahead structural
properties, and behavioral correctness.

---

## Configuration

Everything is driven by YAML. To change parameters, edit the config —
not the code.

```yaml
# configs/eurusd_sma_default.yaml
data:
  source: csv                      # csv | yfinance
  symbol: EURUSD
  timeframe: 1h
  csv_path: tests/fixtures/eurusd_1h_sample.csv

strategy:
  fast_period: 20
  slow_period: 50

risk:
  initial_cash: 10000.0
  risk_per_trade_pct: 0.01          # risk 1% of equity per trade
  stop_loss_pct: 0.005              # 0.5% adverse move
  take_profit_pct: 0.010            # 1.0% target (1:2 RR)

execution:
  spread_pips: 1.0
  slippage_pips: 0.5
  commission_per_lot: 7.0
```

Configs are validated at load time via Pydantic (or a dataclass
fallback). A bad config fails immediately, not three hours into a run.

---

## Project structure

```
bossfx/
├── bossfx/
│   ├── core/              # Events + abstract interfaces + portfolio
│   ├── data/              # CSV, yfinance feeds (MT5 in Phase 5)
│   ├── strategies/        # SMA crossover + online indicators
│   │   └── filters/       # (Phase 2) trend, volatility, session filters
│   ├── risk/              # Percent-risk sizing, DD circuit breaker
│   ├── backtest/          # Event-driven engine + execution simulator
│   ├── analytics/         # Sharpe, Sortino, Calmar, drawdown metrics
│   ├── config/            # YAML -> validated config objects
│   └── utils/             # Structured logging
├── configs/               # User-editable YAMLs
├── tests/                 # 57 tests across 8 files
├── scripts/               # CLI entry points
├── requirements.txt
└── pyproject.toml
```

---

## The three tiers of tests

Every test in the suite defends against one of three failure modes:

1. **Tier 1 — Accounting invariants** (`test_portfolio.py`, `test_events.py`).
   Does one dollar in equal one dollar out? These are the tests where
   money silently disappears if they fail.

2. **Tier 2 — No-look-ahead invariants** (`test_indicators.py`).
   Can any component, under any circumstances, produce a value
   influenced by data it hasn't seen yet? Proven structurally.

3. **Tier 3 — Behavioral correctness** (the rest).
   Does the SMA compute the right number? Does the crossover fire once
   per cross? Does position sizing math check out?

---

## Roadmap

- [x] **Phase 1 — Foundation refactor** *(current)*
  - Event-driven core, abstract interfaces, validated configs,
    realistic execution modeling, 57-test suite, CI-ready.
- [ ] **Phase 2 — Strategy & risk upgrades**
  - Multi-timeframe trend filter (HTF EMA bias)
  - ATR-based volatility filter (skip low-vol regimes)
  - Session filter (trade London/NY overlap only)
  - ATR-based dynamic stops (replace fixed %)
- [ ] **Phase 3 — Realistic backtesting**
  - Walk-forward validation
  - Monte Carlo equity curve confidence intervals
  - Parameter stability tests
- [ ] **Phase 4 — Analytics & reporting**
  - HTML reports (QuantStats-style)
  - Strategy comparison dashboard
  - Trade-level MAE/MFE analytics
- [ ] **Phase 5 — Productization**
  - MT5 live executor
  - Multi-strategy, multi-asset portfolios
  - Optuna parameter optimization
  - Streamlit SaaS dashboard

---

## Design principles

1. **Backtest honesty over backtest beauty.** An 8% return with honest
   assumptions beats a 40% return built on hidden lies.
2. **Every component is replaceable.** The interfaces are contracts;
   implementations plug in. No "god class."
3. **Production-ready means boring.** Defensive, well-logged, well-tested
   code that won't surprise you at 3am when EURUSD spikes on an NFP release.

---

## License

*(Add your chosen license here.)*

---

Built with deliberate intent, not hype. 🏗️
