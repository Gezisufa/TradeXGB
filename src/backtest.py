"""
Vectorised backtest comparing the XGBoost signal strategy to buy-and-hold.

Strategy rules (no leverage, no short):
  - prediction = 1  → hold long (enter/stay long at next open)
  - prediction = 0  → flat (cash, no position)

Transaction costs are NOT modelled.  This is noted explicitly in the README
and inflates strategy returns relative to a live deployment.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
TRADING_DAYS_PER_YEAR = 365  # crypto trades 24/7


def _daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().fillna(0.0)


def run_backtest(
    close: pd.Series,
    predictions: np.ndarray | pd.Series,
    initial_capital: float = 10_000.0,
) -> dict[str, float]:
    """Run a vectorised long/flat backtest and return performance metrics.

    predictions[t] == 1 means hold BTC during day t+1 (signal for the *next* bar).
    Equity curves are saved to results/ as a side-effect.
    """
    close = close.copy()
    preds = pd.Series(np.asarray(predictions), index=close.index)

    daily_ret = _daily_returns(close)

    # The signal on day t applies to the return earned on day t+1
    # (we see today's prediction, hold overnight, sell next close).
    signal = preds.shift(1).fillna(0.0)  # first day: stay flat

    strategy_ret = signal * daily_ret
    bh_ret = daily_ret

    strategy_equity = initial_capital * (1 + strategy_ret).cumprod()
    bh_equity = initial_capital * (1 + bh_ret).cumprod()

    metrics = {
        "strategy": _compute_metrics(strategy_ret, strategy_equity, label="Strategy"),
        "buy_and_hold": _compute_metrics(bh_ret, bh_equity, label="Buy-and-Hold"),
    }

    _plot_equity_curves(strategy_equity, bh_equity)
    _print_comparison(metrics)

    return metrics


def _compute_metrics(returns: pd.Series, equity: pd.Series, label: str) -> dict[str, float]:
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    ann_return = float((1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1)

    excess = returns  # risk-free rate = 0 (standard for crypto)
    sharpe = (
        float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if excess.std() > 0 else 0.0
    )

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_dd = float(drawdown.min())

    # Win rate: fraction of trading days with positive return (when in position)
    in_position = returns != 0.0
    if in_position.sum() > 0:
        win_rate = float((returns[in_position] > 0).mean())
        n_trades = int(in_position.sum())
    else:
        win_rate = 0.0
        n_trades = 0

    result = {
        "total_return": round(total_return, 4),
        "annualised_return": round(ann_return, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "n_active_days": n_trades,
    }
    logger.info("[%s] total_return=%.2f%%  sharpe=%.2f  max_dd=%.2f%%",
                label, total_return * 100, sharpe, max_dd * 100)
    return result


def _plot_equity_curves(strategy: pd.Series, bh: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strategy.index, strategy.values, label="XGBoost Strategy", linewidth=1.5)
    ax.plot(bh.index, bh.values, label="Buy & Hold", linewidth=1.5, linestyle="--")
    ax.set_title("Equity Curve — Strategy vs Buy-and-Hold (test period)")
    ax.set_ylabel("Portfolio value (USD)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "equity_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Equity curve saved to results/equity_curve.png")


def _print_comparison(metrics: dict) -> None:
    print("\n── Backtest Results ────────────────────────────────────────")
    for label, m in metrics.items():
        print(f"\n  {label.replace('_', ' ').title()}")
        print(f"    Total return     : {m['total_return']*100:+.2f}%")
        print(f"    Annualised return : {m['annualised_return']*100:+.2f}%")
        print(f"    Sharpe ratio     : {m['sharpe_ratio']:.2f}")
        print(f"    Max drawdown     : {m['max_drawdown']*100:.2f}%")
        print(f"    Win rate         : {m['win_rate']*100:.1f}%")
        print(f"    Active days      : {m['n_active_days']}")


def save_backtest_metrics(
    metrics: dict,
    existing_path: Path | str | None = None,
) -> None:
    """Merge backtest metrics into results/metrics.json under the 'backtest' key."""
    path = Path(existing_path) if existing_path else RESULTS_DIR / "metrics.json"
    existing: dict = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing["backtest"] = metrics
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info("Backtest metrics merged into %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from data_loader import load_btc
    from features import build_features, chronological_split
    from model import train, evaluate, plot_feature_importance, save_metrics

    df = load_btc()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = chronological_split(X, y)

    clf = train(X_train, y_train)
    model_metrics = evaluate(clf, X_test, y_test)
    plot_feature_importance(clf, list(X.columns))
    save_metrics(model_metrics)

    close_test = df.loc[X_test.index, "Close"]
    bt_metrics = run_backtest(close_test, clf.predict(X_test))
    save_backtest_metrics(bt_metrics)
