"""
Run the Monte Carlo Permutation Test on the real BTC strategy.

Usage:
    python src/run_mcpt.py

This wires the MCPT engine (mcpt_test.py) to the actual TradeXGB pipeline:
for every permuted price series it recomputes features, retrains XGBoost, and
backtests — then reports how often pure-noise data matched the real result.

Keep N_PERMUTATIONS modest (200-300) because XGBoost retrains each loop.
"""
from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_btc
from features import build_features, chronological_split
from model import train
from backtest import run_backtest
from mcpt_test import run_mcpt, plot_mcpt

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
RESULTS_DIR = Path(__file__).parent.parent / "results"

N_PERMUTATIONS = 300   # raise to 1000 if you have time; runtime scales linearly


def strategy_score(ohlcv: pd.DataFrame) -> float:
    """Run the full pipeline on one OHLCV frame and return the strategy's total return.

    Returns -inf for degenerate permutations (e.g. all-NaN indicators) so they
    never count as beating the real result.
    """
    try:
        X, y = build_features(ohlcv)
        if len(X) < 100 or y.nunique() < 2:
            return -np.inf
        X_train, X_test, y_train, _ = chronological_split(X, y)
        if y_train.nunique() < 2:
            return -np.inf

        clf = train(X_train, y_train)
        preds = clf.predict(X_test)
        close_test = ohlcv.loc[X_test.index, "Close"]

        # Silence run_backtest's print/plot side-effects — we only need the scalar
        with contextlib.redirect_stdout(io.StringIO()):
            bt = run_backtest(close_test, preds)
        return float(bt["strategy"]["total_return"])
    except Exception as e:  # noqa: BLE001 — permuted data can be pathological
        logging.debug("pipeline failed on a permutation: %s", e)
        return -np.inf


def main() -> None:
    df = load_btc()
    print(f"Loaded {len(df)} bars: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Running MCPT with {N_PERMUTATIONS} permutations (XGBoost retrains each time)…")
    print("This takes a few minutes.\n")

    real_score, p_value, permuted = run_mcpt(
        df, strategy_score, n_permutations=N_PERMUTATIONS, seed=42,
    )

    # Clean stats (ignore any -inf from degenerate permutations)
    finite = permuted[np.isfinite(permuted)]
    print("\n────────────────  MCPT RESULT  ────────────────")
    print(f"  Real strategy total return : {real_score:+.4f}")
    print(f"  Permuted mean / median     : {finite.mean():+.4f} / {np.median(finite):+.4f}")
    print(f"  Permuted 95th percentile   : {np.percentile(finite, 95):+.4f}")
    print(f"  p-value                    : {p_value:.4f}")
    print("───────────────────────────────────────────────")
    if p_value > 0.05:
        print("  → NOT statistically significant. The result is consistent with")
        print("    random chance — no demonstrable edge. (Expected for daily crypto.)")
    else:
        print("  → Statistically significant at 5%. Scrutinise hard for leakage")
        print("    before believing it — real edge on daily direction is rare.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_mcpt(real_score, finite, p_value, save_path=RESULTS_DIR / "mcpt_histogram.png")
    print(f"\nHistogram saved to results/mcpt_histogram.png")


if __name__ == "__main__":
    main()
