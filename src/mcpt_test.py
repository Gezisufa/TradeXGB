"""
Monte Carlo Permutation Test (MCPT) for the BTC direction strategy.

WHY THIS EXISTS
---------------
A backtest result on a single slice of history can look good (or bad) purely by
chance. MCPT answers one question rigorously:

    "Could my strategy's result have arisen from random noise,
     or does it reflect a real, repeatable edge?"

HOW IT WORKS (say this in your own words at interview)
------------------------------------------------------
1. We take the REAL price series and run the full pipeline
   (features -> train XGBoost -> backtest), recording the real result.
2. We then build many PERMUTED price series. Each permutation keeps the
   *distribution* of bar-to-bar moves identical (same volatility, same
   building blocks) but SHUFFLES their time order, destroying any genuine
   temporal pattern the model could exploit.
3. On each permuted series we rerun the SAME pipeline and record the result.
   This builds a distribution of results "under the null hypothesis of no
   real signal".
4. p-value = fraction of permuted runs that did AT LEAST AS WELL as the real
   run. A high p-value (e.g. 0.40) means 40% of pure-noise series matched our
   result -> the result is NOT statistically significant.

The permutation engine is adapted from neurotrader888's MCPT framework
(github.com/neurotrader888/mcpt). Using a vetted, well-known method here is a
deliberate engineering choice: reach for the right proven tool rather than
reinvent a statistical test from scratch.

NOTE ON METHODOLOGY
-------------------
This is an *in-sample* style MCPT: on each permutation we retrain the model and
score it, mirroring the reference framework. Because XGBoost retrains each loop,
keep n_permutations modest (200-500) for daily data; it runs in minutes.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Adapted from neurotrader888/mcpt (bar_permute.py) to TradeXGB's OHLCV format.
def get_permutation(
    ohlcv: pd.DataFrame,
    start_index: int = 0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Return an OHLCV DataFrame with permuted bar order but identical return distribution.

    Works in log space on relative intrabar components (gap, body, wicks) so the
    statistical properties of individual bars are preserved while temporal patterns
    are destroyed. Volume is shuffled with the same permutation as price moves to
    keep the intrabar volume/move relationship intact.
    """
    rng = np.random.default_rng(seed)

    cols = ["Open", "High", "Low", "Close"]
    n_bars = len(ohlcv)
    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    log_bars = np.log(ohlcv[cols].to_numpy())  # shape (n_bars, 4)
    log_open, log_high, log_low, log_close = (
        log_bars[:, 0], log_bars[:, 1], log_bars[:, 2], log_bars[:, 3],
    )

    r_open = np.empty(n_bars)
    r_open[1:] = log_open[1:] - log_close[:-1]  # gap: open relative to prev close
    r_open[0] = 0.0
    r_high = log_high - log_open   # upper wick
    r_low = log_low - log_open     # lower wick
    r_close = log_close - log_open # body

    idx = np.arange(perm_n)
    # perm1: shuffle intrabar shape (H/L/C + volume) together
    # perm2: shuffle open-gaps separately — matches the reference design
    perm1 = rng.permutation(idx)
    perm2 = rng.permutation(idx)

    r_high_s = r_high[perm_index:][perm1]
    r_low_s = r_low[perm_index:][perm1]
    r_close_s = r_close[perm_index:][perm1]
    r_open_s = r_open[perm_index:][perm2]

    # Volume follows perm1 so the within-bar volume/price-move pairing is intact
    vol = ohlcv["Volume"].to_numpy()
    vol_perm = vol.copy()
    vol_perm[perm_index:] = vol[perm_index:][perm1]

    perm_bars = np.zeros((n_bars, 4))
    perm_bars[:perm_index] = log_bars[:perm_index]  # warm-up bars unchanged

    for i in range(perm_index, n_bars):
        k = i - perm_index
        new_open = perm_bars[i - 1, 3] + r_open_s[k]   # prev close + gap
        perm_bars[i, 0] = new_open
        perm_bars[i, 1] = new_open + r_high_s[k]
        perm_bars[i, 2] = new_open + r_low_s[k]
        perm_bars[i, 3] = new_open + r_close_s[k]

    perm_prices = np.exp(perm_bars)
    out = pd.DataFrame(perm_prices, index=ohlcv.index, columns=cols)
    out["Volume"] = vol_perm
    return out


def run_mcpt(
    ohlcv: pd.DataFrame,
    run_pipeline: Callable[[pd.DataFrame], float],
    n_permutations: int = 300,
    start_index: int = 0,
    seed: int | None = 42,
) -> tuple[float, float, np.ndarray]:
    """Run a Monte Carlo Permutation Test and return (real_score, p_value, permuted_scores).

    run_pipeline must encapsulate the ENTIRE strategy (features → train → backtest)
    and return a single scalar where higher = better.
    p_value = fraction of permuted runs that matched or beat the real result.
    """
    real_score = run_pipeline(ohlcv)
    logger.info("Real score: %.4f", real_score)

    permuted_scores = np.empty(n_permutations)
    perm_better = 1  # count the real run itself — standard MCPT p-value convention

    for i in range(n_permutations):
        perm = get_permutation(ohlcv, start_index=start_index, seed=None if seed is None else seed + i + 1)
        score = run_pipeline(perm)
        permuted_scores[i] = score
        if score >= real_score:
            perm_better += 1

    p_value = perm_better / (n_permutations + 1)
    logger.info("MCPT p-value: %.4f  (%d permutations)", p_value, n_permutations)
    return real_score, p_value, permuted_scores


def plot_mcpt(real_score: float, permuted_scores: np.ndarray, p_value: float, save_path=None):
    """Histogram of permuted scores with the real score marked."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(permuted_scores, bins=40, color="#4C72B0", alpha=0.8, label="Permutations (null)")
    ax.axvline(real_score, color="#C44E52", linewidth=2, label="Real result")
    ax.set_xlabel("Strategy score")
    ax.set_ylabel("Count")
    ax.set_title(f"Monte Carlo Permutation Test — p-value = {p_value:.3f}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
